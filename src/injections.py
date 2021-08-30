import time
import torch.nn as nn
from util import *
from tqdm import tqdm
from goldeneye import goldeneye

sys.path.append("./pytorchfi")
from error_models_num_sys import (
    num_fp32,
    num_fp16,
    num_bfloat16,
    num_fixed_pt,
)


def rand_neurons_batch(pfi_model, layer, shape, maxval, batchsize):
    dim = len(shape)
    batch, layerArr, dim1, dim2, dim3, value = ([] for i in range(6))

    for i in range(batchsize):
        batch.append(i)
        layerArr.append(layer)
        value.append(random_value(-1.0 * maxval, maxval))

        dim1val = random.randint(0, shape[0]-1)
        dim1.append(dim1val)
        if dim >= 2:
            dim2val = random.randint(0, shape[1]-1)
            dim2.append(dim2val)
        else:
            dim2.append(None)

        if dim >= 3:
            dim3val = random.randint(0, shape[2]-1)
            dim3.append(dim3val)
        else:
            dim3.append(None)

    return pfi_model.declare_neuron_fi(
        batch=batch,
        layer_num=layerArr,
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
        value=value,
    )


def quantize(module, input, output):
    # NOTE: pytorch recommends not using apply_ in
    # situations where high performance is needed.
    # The function apply_ seems to only support cpu tensors (?)
    return output.apply_(lambda val: num_fp32().quantize(val))


if __name__ == "__main__":

    # Read in cmd line args
    check_args(sys.argv[1:])
    if getDebug():
        printArgs()

    sys.path.append(getOutputDir() + "../src/pytorchfi") #when calling from ./scripts/
    from pytorchfi.core import fault_injection
    from pytorchfi.error_models import *

    inj_per_layer = getInjections()
    assert inj_per_layer != -1, "The number of injections is not valid (-1)"

    # common variables
    name = getDNN() + "_" + getDataset() + "_" + getPrecision()
    range_path = getOutputDir() + "/networkRanges/" + name + "/"
<<<<<<< HEAD
    profile_path = getOutputDir() + "/networkProfiles/" + name + "/"
    data_susbet_path = getOutputDir() + "/data_subset/" + name + "/"
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

    # TODO
    # quantization hooks go here

    # # register forward hook to the model
    # for param in model.modules():
    #     if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
    #         param.register_forward_hook(quantize)

    # init PyTorchFI
    baseC = 3
    if "IMAGENET" in getDataset():
        baseH = 224
        baseW = 224
    elif "CIFAR" in getDataset():
        baseH = 224
        baseW = 224

    pfi_model = fault_injection(
        model,
        getBatchsize(),
        input_shape=[baseC, baseH, baseW],
        layer_types=[nn.Conv2d, nn.Linear],
        use_cuda=True,
    )

    if getDebug():
        print(pfi_model.print_pytorchfi_layer_summary())

    assert pfi_model.get_total_layers() == total_layers
    shapes = pfi_model.get_output_size()

    # ERROR INJECTION CAMPAIGN
    start_time = time.time()
    for currLayer in tqdm(range(pfi_model.get_total_layers()), desc="Layers"):
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

            # injection locations
            inj_model = goldeneye(
                model,
                getBatchsize(),
                input_shape=[baseC, baseH, baseW],
                layer_types=[nn.Conv2d, nn.Linear],
                use_cuda=True,
                num_sys=num_bfloat16(),
                quant=True,
                layer_max=[],
                inj_order=1,
            )

            # rand_neurons_batch(
            #     pfi_model, currLayer, currShape, maxVal, getBatchsize()
            # )

            # perform inference
            output_inj = inj_model(images)
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
            print("")
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
