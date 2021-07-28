import torch.nn as nn
from util import *
from pytorchfi.pytorchfi.core import fault_injection

if __name__ == '__main__':

    # Read in cmd line args
    check_args(sys.argv[1:])
    if getDebug(): printArgs()

    inj_per_layer = getInjections()
    assert inj_per_layer != -1, "The number of injections is not valid (-1)"

    # common variables
    name = getDNN() + "_" + getDataset() + "_" + getPrecision()
    range_path = getOutputDir() + "/networkRanges/" + name + "/"
    profile_path= getOutputDir() + "/networkProfiles/" + name + "/"
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
    good_img_set = load_file(data_susbet_path + image_set)




