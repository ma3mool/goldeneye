import time
from util import *
from PIL import Image, ImageDraw
import torch.nn as nn
from tqdm import tqdm
from goldeneye import goldeneye
from torchvision.transforms.functional import to_pil_image
from profiling import calculate_precision_recall_f1,check_labels_tensors

# from num_sys_class import *
# sys.path.append("./pytorchfi")

def rand_neurons_batch(pfi_model, layer, shape, maxval, batchsize, function=-1):
    dim = len(shape)
    batch, layerArr, dim1, dim2, dim3, value = ([] for i in range(6))

    for i in range(batchsize):
        batch.append(i)
        layerArr.append(layer)
        if function == -1:
            value.append(random_value(-1.0 * maxval, maxval))

        dim1val = random.randint(0, shape[0] - 1)
        dim1.append(dim1val)
        if dim >= 2:
            dim2val = random.randint(0, shape[1] - 1)
            dim2.append(dim2val)
        else:
            dim2.append(None)

        if dim >= 3:
            dim3val = random.randint(0, shape[2] - 1)
            dim3.append(dim3val)
        else:
            dim3.append(None)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layerArr,
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
        function=function,
    )

if __name__ == "__main__":

    # Read in cmd line args
    check_args(sys.argv[1:])
    printArgs()
    if getDebug():
        printArgs()

    # sys.path.append(getOutputDir() + "../src/pytorchfi")  # when calling from ./scripts/
    # from pytorchfi.core import fault_injection
    # from pytorchfi.neuron_error_models import *

    inj_per_layer = getInjections()
    assert inj_per_layer != -1, "The number of injections is not valid (-1)"

    # common variables
    range_name = getDNN() + "_" + getDataset()
    range_path = getOutputDir() + "/networkRanges/" + range_name + "/"

    if getFormat() == "INT":
        format = "INT"
        quant_en = True
        bitwidth_fp = 32
    else:
        format = getFormat()
        bitwidth_fp = getBitwidth()
        quant_en = False

    name = getDNN() + "_" + getDataset() + "_real" + getPrecision() + "_sim" + format + "_bw" + str(bitwidth_fp) \
           + "_r" + str(getRadix()) + "_bias" + str(getBias())

    profile_path = getOutputDir() + "/networkProfiles/" + name + "/"
    data_subset_path = getOutputDir() + "/data_subset/" + name + "/"
    out_path = getOutputDir() + "/injections/" + name + "/"
    out_saved_imgs_path = getOutputDir() + "/injections/" + name + "/inj_output_imgs"

    if os.path.exists(out_saved_imgs_path):
        pass
    else:
        os.mkdir(out_saved_imgs_path)

    image_set = ""

    if getTraining_en():
        image_set = "rank_set_good"
        out_path += "training/"
    else:
        image_set = "test_set_good"
        out_path += "testing/"

    # load important info: ranges, mapping, good images
    ranges = load_file(range_path + "ranges_trainset_layer")
    good_img_set = load_file(data_subset_path + image_set)

    # constants
    total_layers = len(ranges)
    total_inferences = getInjections() * total_layers

    # Use custom data loader
    dataiter = load_custom_dataset(
        getDNN(),
        getDataset(),
        getBatchsize(),
        good_img_set,
        total_inferences,
        workers=getWorkers(),
    )
    model = getNetwork(getDNN(), getDataset())
    criterion = torchvision.models.detection.fasterrcnn_resnet50_fpn().cuda()
    criterion.train()
    img_ids = 0
    #criterion = nn.CrossEntropyLoss(reduction="none")

    if getCUDA_en():
        model = model.cuda()
    if getPrecision() == "FP16":
        model = model.half()
    print(" Dataset & Model loaded....")
    model.eval()
    torch.no_grad()

    # init PyTorchFI
    baseC = 3
    if "IMAGENET" in getDataset():
        baseH = 224
        baseW = 224
    
    elif "CIFAR" in getDataset():
        baseH = 32
        baseW = 32

    elif "COCO" in getDataset():
        baseH = 480
        baseW = 640

    exp_bits = getBitwidth() - getRadix() - 1  # also INT for fixed point
    mantissa_bits = getRadix()  # also FRAC for fixed point

    goldeneye_model = goldeneye(
        model,
        getBatchsize(),
        input_shape=[baseC, baseH, baseW],
        layer_types=[nn.Conv2d, nn.Linear],
        use_cuda=getCUDA_en(),

        # number format
        signed=True,
        num_sys=getNumSysName(getFormat(),
                              bits=bitwidth_fp,
                              radix_up=exp_bits,
                              radix_down=mantissa_bits,
                              bias=getBias()),

        # quantization
        quant=quant_en,
        layer_max=ranges,
        bits=getBitwidth(),
        qsigned=True,

        inj_order=getInjectionsLocation(),
    )

    if getDebug():
        print(goldeneye_model.print_pytorchfi_layer_summary())

    assert goldeneye_model.get_total_layers() == total_layers
    shapes = goldeneye_model.get_output_size()

    # ERROR INJECTION CAMPAIGN
    start_time = time.time()
    for currLayer in tqdm(range(goldeneye_model.get_total_layers()), desc="Layers"):
        layerInjects = []

        maxVal = ranges[currLayer]
        currShape = shapes[currLayer][1:]

        pbar = tqdm(total=inj_per_layer, desc="Inj per layer")
        samples = 0
        while samples < inj_per_layer:
            pbar.update(samples)

            # prep images
            for images, labels, index in dataiter:
                images = list(img.cuda() for img in images)
                labels = [{k: v.cuda() for k, v in t.items()} for t in labels]
                
                if getPrecision() == "FP16":
                    images = images.half()

            # inj_model = random_neuron_single_bit_inj_batched(pfi_model, ranges)
            # inj_model_locations = random_neuron_inj_batched(pfi_model,
            #                                         min_val= abs(ranges[currLayer]) * -1,
            #                                         max_val=abs(ranges[currLayer]),
            #                                       )
            # injection locations
            with torch.no_grad():
                inf_model = rand_neurons_batch(goldeneye_model,
                                           currLayer,
                                           currShape,
                                           maxVal,
                                           getBatchsize(),
                                           function=goldeneye_model.apply_goldeneye_transformation
                                           )
            # injection model
            # inf_model = goldeneye.declare_neuron_fi(batch=batch_inj,
            #                                         layer=layer_inj,
            #                                         dim1=dim1_inj,
            #                                         dim2=dim2_inj,
            #                                         dim3=dim3_inj,
            #                                         function=goldeneye.apply_goldeneye_transformation,
            #                                         )

            # perform inference
                output_inj = inf_model(images)
                out_loss = criterion(images,labels)
                cls_loss = out_loss['loss_classifier'].item()
                bbox_loss = out_loss['loss_box_reg'].item()
                obj_loss = out_loss['loss_objectness'].item()
                rpn_loss = out_loss['loss_rpn_box_reg'].item()
                out_inj_lossess = [cls_loss, bbox_loss, obj_loss, rpn_loss]

            for idx, img in enumerate(images):
                try:
                    img = to_pil_image(img)
                    draw = ImageDraw.Draw(img)

                    inj_pred_bboxes = output_inj[idx]['boxes'].int().cpu()
                    inj_pred_labels = output_inj[idx]['labels'].cpu()
                    inj_pred_box_confs = output_inj[idx]['scores']
                    gr_label = labels[idx]['labels'].cpu()
                    inj_correct_boxes = check_labels_tensors(gr_label.cpu(), inj_pred_labels.cpu())[0]
                    inj_wrong_boxes = check_labels_tensors(gr_label.cpu(), inj_pred_labels.cpu())[1]

                    inj_precisions= calculate_precision_recall_f1(output_inj[idx]['boxes'],labels[idx]['boxes'], 0.60)

                    for box, inj_pd_label, inj_score, in zip(inj_pred_bboxes, inj_pred_labels, inj_pred_box_confs):
                        if inj_score > 0.4:
                            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="blue", width=4)
                            draw.text((box[0], box[1]), f"{inj_pd_label}: {inj_score:.2f}", fill="red")
                            img.save(f"{out_saved_imgs_path}/output_{index[idx]}_{str(currLayer)}{inj_score:.2f}.png")
                except:
                    continue

                layerInjects.append(
                    (
                        index,
                        inj_pred_bboxes,
                        inj_pred_labels,
                        inj_pred_box_confs,
                        inj_precisions,
                        inj_correct_boxes,
                        inj_wrong_boxes,
                        out_inj_lossess,
                    )
                )
            samples += getBatchsize()
            torch.cuda.empty_cache()
            # print("")
        pbar.close()

        fileName = "layer" + str(currLayer)
        save_data(out_path, fileName, layerInjects)

    end_time = time.time()

    print("========================================================")
    print(name)
    print("Set: ", "Rank Set" if getTraining_en() else "Test Set")
    # print("Quantization:", getQuantize_en())
    # print("Error Model:", "Random Value" if not getSingleBitFlip_en() else "Single bit flip")
    print("Total Error Injections:", total_inferences)
    print("Total Runtime: {}".format(end_time - start_time) + " seconds")
    print("========================================================")
