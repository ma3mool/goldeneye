from util import *
import torch.nn as nn
from tqdm import tqdm


activations = []
def save_activations(module, input, output):
    activations.append(output)

def gather_min_max_per_layer(model, data_iter, batch_size, precision="FP16", cuda_en=True, debug=False, verbose=False):
    global activations
    layer_max = torch.Tensor([]).cuda().half()  # ADDED
    layer_min = torch.Tensor([]).cuda().half()  # ADDED

    # register forward hook to the model
    handles = []
    for param in model.modules():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
            handles.append(param.register_forward_hook(save_activations))

    # main loops to gather ranges
    processed_elements = 0
    batch_num = 0

    for input_data in tqdm(data_iter):

        # prepare the next batch for inference
        images, labels = input_data
        if cuda_en:
            images = images.cuda()
            labels = labels.cuda()
        if precision == "FP16": images = images.half()

        activations = []  # reset before every inference
        model(images)  # run an inference

        # Range gathering: iterate through each layer

        min_vals = (
            torch.Tensor(list(map(lambda layer: layer.min().item(), activations)))
            .cuda()
            .half()
        )
        max_vals = (
            torch.Tensor(list(map(lambda layer: layer.max().item(), activations)))
            .cuda()
            .half()
        )
        if batch_num == 0:
            layer_max = max_vals
            layer_min = min_vals
        else:
            layer_max = torch.max(layer_max, max_vals)
            layer_min = torch.min(layer_min, min_vals)

        processed_elements += len(labels)
        batch_num += 1
        torch.cuda.empty_cache()

    # remove hooks
    for i in range(len(handles)):
        handles[i].remove()
    del activations

    actual_max = torch.max(torch.abs(layer_min), torch.abs(layer_max))
    return layer_min, layer_max, actual_max

if __name__ == '__main__':

    # read in cmd line args
    check_args(sys.argv[1:])
    if getDebug(): printArgs()

    # common variables
    name = getDNN() + "_" + getDataset() + "_" + getPrecision()
    out_path = getOutputDir() + "/networkRanges/" + name + "/"

    # load data and model
    dataiter = load_dataset(getDataset(), getBatchsize(), workers = getWorkers(), training=True, include_id=False)
    model = getNetwork(getDNN(), getDataset())
    model.eval()
    torch.no_grad()

    # Profile and collect ranges
    layer_min, layer_max, actual_max = gather_min_max_per_layer(
        model,
        dataiter,
        getBatchsize(),
        cuda_en=getCUDA_en(),
    )
    ranges = actual_max.cpu().numpy().tolist()

    # save result
    save_data(out_path, "ranges_trainset_layer", ranges)

    # save CSV
    f = open(out_path + "ranges_trainset_layer.csv", "w+")
    for i in range(len(ranges)):
        outputString = "%d, %f\n" % (i, ranges[i])
        f.write(outputString)
    f.close()