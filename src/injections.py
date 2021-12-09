import time
import torch.nn as nn

# import pytorchfi.pytorchfi.error_models
from util import *
from tqdm import tqdm
from goldeneye import goldeneye

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
    criterion = nn.CrossEntropyLoss(reduction="none")

    if getCUDA_en():
        model = model.cuda()
    if getPrecision() == "FP16":
        model = model.half()
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

    exp_bits = getBitwidth() - getRadix() - 1  # also INT for fixed point
    mantissa_bits = getRadix()  # also FRAC for fixed point
    goldeneye = goldeneye(
        model,
        getBatchsize(),
        layer_types=[nn.Conv2d, nn.Linear],
        use_cuda=getCUDA_en(),

        # number format
        signed=True,
        num_sys=getNumSysName(getFormat(),
                              bits=getBitwidth(),
                              radix_up=exp_bits,
                              radix_down=mantissa_bits,
                              bias=getBias()),

        # num_sys=getNumSysName(getFormat()),

        # quantization
        quant=getQuantize_en(),
        layer_max=ranges,
        bits=getBitwidth(),
        qsigned=True,

        inj_order = getInjectionsLocation(),
    )

    if getDebug():
        print(goldeneye.print_pytorchfi_layer_summary())

    assert goldeneye.get_total_layers() == total_layers
    shapes = goldeneye.get_output_size()

    # ERROR INJECTION CAMPAIGN
    start_time = time.time()
    for currLayer in tqdm(range(goldeneye.get_total_layers()), desc="Layers"):
        layerInjects = []

        maxVal = ranges[currLayer]
        currShape = shapes[currLayer][1:]

        pbar = tqdm(total=inj_per_layer, desc="Inj per layer")
        samples = 0
        while samples < inj_per_layer:
            pbar.update(samples)

            # prep images
            images, labels, img_ids, index = dataiter.next()
            if getCUDA_en():
                labels = labels.cuda()
                images = images.cuda()
            if getPrecision() == "FP16":
                images = images.half()


            # inj_model = random_neuron_single_bit_inj_batched(pfi_model, ranges)
            # inj_model_locations = random_neuron_inj_batched(pfi_model,
            #                                         min_val= abs(ranges[currLayer]) * -1,
            #                                         max_val=abs(ranges[currLayer]),
            #                                       )

            # injection locations
            inf_model = rand_neurons_batch(goldeneye,
                                           currLayer,
                                           currShape,
                                           maxVal,
                                           getBatchsize(),
                                           function=goldeneye.apply_goldeneye_transformation
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
            output_argmax = torch.argmax(output_inj, dim=1)
            output_inj_loss = criterion(output_inj, labels)

            # save results
            layerInjects.append(
                (
                    index.tolist(),
                    output_argmax.data.tolist(),
                    output_inj_loss.tolist(),
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
