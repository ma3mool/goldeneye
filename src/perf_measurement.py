from util import *
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from goldeneye import *
import time as time
import statistics

if __name__ == '__main__':

    MAX_RUNS=5

    # read in cmd line args
    check_args(sys.argv[1:])
    if getDebug(): printArgs()

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
    # if getQuantize_en(): name += "_" + "quant"
    out_path = getOutputDir() + "/networkProfiles/" + name + "/"
    subset_path = getOutputDir() + "/data_subset/" + name + "/"

    # get ranges
    ranges = load_file(range_path + "ranges_trainset_layer")

    # load data and model
    dataiter = load_dataset(getDataset(), getBatchsize(), workers = getWorkers())
    model = getNetwork(getDNN(), getDataset())
    model.eval()
    torch.no_grad()

    exp_bits = getBitwidth() - getRadix() - 1  # also INT for fixed point
    mantissa_bits = getRadix() #getBitwidth() - exp_bits - 1  # also FRAC for fixed point

    # no injections during profiling
    assert(getInjections() == -1)
    assert(getInjectionsLocation() == 0)

    goldeneye_model = goldeneye(
        model,
        getBatchsize(),
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

    input_data = dataiter.next()

    inf_model = goldeneye_model.declare_neuron_fi(function=goldeneye_model.apply_goldeneye_transformation)

    # prepare the next batch for inference
    images, labels, img_ids, index = input_data
    if getPrecision() == 'FP16': images = images.half()
    if getCUDA_en():
        images = images.cuda()
        labels = labels.cuda()

    # warm-up
    with torch.no_grad():
        for runs in range(32):
            output=inf_model(images)
        torch.cuda.empty_cache()

    times = []
    with torch.no_grad():
        for runs in range(MAX_RUNS):
            start_time = time.time()
            output = inf_model(images) # run an inference
            torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            torch.cuda.empty_cache()

    ave_time = sum(times)/MAX_RUNS
    res = statistics.pstdev(times)

    print("Name: ", name)
    print("Ave Time: ", ave_time)
    print("STD Time: ", res)
