from util import *
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from goldeneye import *


# iterates through through test dataset and returns measured accuracy on particular numsys
@torch.no_grad()
def test_accuracy(goldeneye, data_iter, cuda_en=True, precision='FP16', verbose=False, debug=False):
    if debug:
        processed_elements = 0
        max_elements = 300

    counter = 0
    correct = 0
    measured_top1conf = 0.0
    measured_top2diff = 0.0
    inf_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction='none')

    for input_data in tqdm(data_iter):
        if debug and processed_elements >= max_elements: break

        inf_model = goldeneye.declare_neuron_fi(function=goldeneye.apply_goldeneye_transformation)

        # prepare the next batch for inference
        images, labels, img_ids, index = input_data
        if precision == 'FP16': images = images.half()
        if cuda_en:
            images = images.cuda()
            labels = labels.cuda()

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
                correct += 1
                measured_top1conf += conf[img].item() * 100
                measured_top2diff += diff_top2(top2[img])
            inf_loss += output_loss[img].item()

        if debug:
            processed_elements += len(labels)
        # torch.cuda.empty_cache()

    ave_correct = correct / counter
    ave_top1conf = measured_top1conf / counter
    ave_top2diff = measured_top2diff / counter
    ave_loss = inf_loss / counter

    return ave_correct, ave_top1conf, ave_top2diff, ave_loss

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

    name = getDNN() + "_" + getDataset() + "_real" + getPrecision() + "_sim" + getFormat()
    if getQuantize_en(): name += "_" + "quant"
    out_path = getOutputDir() + "/networkProfiles/" + name + "/"
    subset_path = getOutputDir() + "/data_subset/" + name + "/"

    # get ranges
    ranges = load_file(range_path + "ranges_trainset_layer")

    # load data and model
    dataiter = load_dataset(getDataset(), getBatchsize(), workers = getWorkers())
    model = getNetwork(getDNN(), getDataset())
    model.eval()
    torch.no_grad()

    FLOAT_IGNORE = 8
    num_formats = ["fp_n", "fixedpt", "block_fp", "adaptive_fp"]
    # quant_formats = {}
    bit_widths = list(reversed(range(2, 32)))

    count = 0
    for num_format in num_formats:
        # for quant_format in quant_formats:
        for bit_width in bit_widths:
            # severe lack of precision in floats below FLOAT_IGNORE. Skip these
            if "fp" in num_format and bit_width < FLOAT_IGNORE:
                continue

            # Sweep radix point
            for radix in range(1, bit_width):
                exp_bits = bit_width - radix - 1            # also INT for fixed point
                mantissa_bits = bit_width - exp_bits - 1    # also FRAC for fixed point

                print("[%s] %d bits: (1, %d, %d)" %(num_format, bit_width, exp_bits, mantissa_bits))
                count += 1
    print("Count:", count)
                goldeneye_model = goldeneye(
                    model,
                    getBatchsize(),
                    layer_types=[nn.Conv2d, nn.Linear],
                    use_cuda=getCUDA_en(),
                    layer_max=ranges,

                    # number system
                    num_sys=getNumSysName(num_format),
                    signed=True,
                    bits=8,
                    radix=radix,

                    # quantization
                    quant=getQuantize_en(),
                    quant_numsys=getNumSysName(num_format),
                    qsigned=True,
                    qbits=8,
                    qradix=radix,

                    inj_order=False,
                )

                # Golden data gathering
                accuracy, top1conf, top2diff, ave_loss = test_accuracy(goldeneye_model, dataiter, \
                        getBatchsize(), precision=getPrecision(), verbose=getVerbose(), debug=getDebug())

    # TODO end for

    # Golden data gathering
    # output_name = "golden_data"
    # save_data(out_path, output_name, golden_data)
    # df = save_data_df(out_path, output_name, golden_data)

    # Print Summary Statistics
    # summaryDetails = ""
    # summaryDetails += "===========================================\n"
    # summaryDetails += "%s\n" % (name)
    # summaryDetails += "Accuracy: \t%0.2f%%\n" % (accuracy * 100)
    # summaryDetails += "Ave Conf: \t%0.2f%%\n" % (top1conf)
    # summaryDetails += "Ave Top2Diff: \t%0.2f%%\n" % (top2diff)
    # summaryDetails += "Ave Loss: \t%0.2f\n" % (ave_loss)
    # summaryDetails += "===========================================\n"
    #
    # if getVerbose():
    #     print(summaryDetails)