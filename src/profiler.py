from util import *
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from goldeneye import *


# goes through dataset and finds correctly classified images, and corresponding data
@torch.no_grad()
def gather_golden(goldeneye, data_iter, cuda_en=True, precision='FP32', verbose=False, debug=False):
    golden_data = {}
    processed_elements = 0
    good_imgs = 0
    bad_imgs = 0

    criterion = nn.CrossEntropyLoss(reduction='none')
    counter = 0
    for input_data in tqdm(data_iter):
        inf_model = goldeneye.declare_neuron_fi(function=goldeneye.apply_goldeneye_transformation)
        # if debug:
        #     if processed_elements >= max_elements:
        #         break

        # prepare the next batch for inference
        images, labels, img_ids, index = input_data
        if precision == 'FP16': images = images.half()
        if cuda_en:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            output = inf_model(images) # run an inference
            output_loss = criterion(output, labels)

            # get argmax and conf
            output_soft = F.softmax(output,dim=1)
            conf, argMax = output_soft.max(1)

            # get top2diff
            top2 = torch.topk(output_soft, k=2, dim=1)[0]

        for img in range(len(images)):
            counter += 1
            inf_label = argMax[img].item()

            if inf_label == labels[img].item():
                correct_inf = True
                good_imgs += 1
            else:
                correct_inf = False
                bad_imgs += 1

            inf_conf = conf[img].item() * 100
            inf_top2diff = diff_top2(top2[img])
            inf_loss = output_loss[img].item()
            # img_tuple = (correct_inf, inf_label, inf_conf, inf_top2diff, inf_loss)
            img_tuple = (labels[img].item(), inf_label, inf_conf, inf_top2diff, inf_loss)
            img_id = index[img].item()

            assert(img_id not in golden_data) # we shouldn't have duplicates in the golden data
            golden_data[img_id] = img_tuple

        processed_elements += len(labels)
        torch.cuda.empty_cache()

    total_imgs = good_imgs + bad_imgs
    return golden_data, good_imgs, bad_imgs, total_imgs

def save_data_df(path, file_name, data):
    if not os.path.exists(path):
        os.makedirs(path)
    output = path + file_name
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.columns = ['img_id', 'correct_inf', 'inf_label', 'inf_conf', 'inf_top2diff', 'inf_loss']
    df.to_pickle(output + ".df")
    df.to_csv(output + ".csv", index=False)

    return df


if __name__ == '__main__':

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

    # init PyTorchFI
    baseC = 3
    if "IMAGENET" in getDataset():
        baseH = 224
        baseW = 224
    elif "CIFAR" in getDataset():
        baseH = 32
        baseW = 32


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
        layer_ranges=ranges,
        bits=getBitwidth(),
        qsigned=True,

        inj_order=getInjectionsLocation(),
    )

    # Golden data gathering
    golden_data, good_imgs, bad_imgs, total_imgs = gather_golden(goldeneye_model, dataiter, \
            getBatchsize(), precision=getPrecision(), verbose=getVerbose(), debug=getDebug())

    # Golden data gathering
    output_name = "golden_data"
    save_data(out_path, output_name, golden_data)
    df = save_data_df(out_path, output_name, golden_data)

    # Print Summary Statistics
    summaryDetails = ""
    summaryDetails += "===========================================\n"
    summaryDetails += "%s\n" % (name)
    summaryDetails += "Accuracy: \t%0.2f%%\n" % (good_imgs / total_imgs * 100.0)
    summaryDetails += "Ave Loss: \t%0.2f\n" % (df["inf_loss"].mean())
    summaryDetails += "Ave Conf: \t%0.2f%%\n" % (df["inf_conf"].mean())
    summaryDetails += "Ave Top2Diff: \t%0.2f%%\n" % (df["inf_top2diff"].mean())
    summaryDetails += "===========================================\n"

    # save stats
    stats_file = open(out_path + "stats.txt", "w+")
    n = stats_file.write(summaryDetails)
    stats_file.close()


    if getVerbose():
        print(summaryDetails)
        # print("===========================================")
        # print(name)
        # print("Accuracy: \t%0.2f%%" %(good_imgs / total_imgs * 100.0))
        # print("Ave Loss: \t%0.2f" %(df["inf_loss"].mean()))
        # print("Ave Conf: \t%0.2f%%" %(df["inf_conf"].mean()))
        # print("Ave Top2Diff: \t%0.2f%%" %(df["inf_top2diff"].mean()))
        # print("===========================================")
