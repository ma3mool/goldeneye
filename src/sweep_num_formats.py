from util import *
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from goldeneye import *


# iterates through through test dataset and returns measured accuracy on particular numsys
@torch.no_grad()
def test_accuracy(goldeneye, data_iter, cuda_en=True, precision='FP32', verbose=False, debug=False):
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

    ave_correct = correct / counter * 100.0
    ave_top1conf = measured_top1conf / counter
    ave_top2diff = measured_top2diff / counter
    ave_loss = inf_loss / counter

    return ave_correct, ave_top1conf, ave_top2diff, ave_loss

def run_goldeneye_profile(model, dataset, batchsize, workers,
                          num_format, sign_numsys, bit_width, radix, bias,
                          ranges, quant_en, qsigned, qbits, verbose=False):

    exp_bits = bit_width - radix - 1  # also INT for fixed point
    mantissa_bits = bit_width - exp_bits - 1  # also FRAC for fixed point

    # if verbose:
    #     print("[%s] %d bits: (1, %d, %d), Quant: %s, %d," %(num_format, bit_width, exp_bits, mantissa_bits, quant_en, qbits))
    # return (0, 0, 0, 0)

    dataiter = load_dataset(dataset, batchsize, workers=workers)

    goldeneye_model = goldeneye(
        model,
        batchsize,
        layer_types=[nn.Conv2d, nn.Linear],

        use_cuda=True,

        # number system
        num_sys=getNumSysName(num_format,
                              bits=bit_width,
                              radix_up=exp_bits,
                              radix_down=mantissa_bits,
                              bias=bias),
        signed=sign_numsys,

        # quantization
        layer_max=ranges,
        quant=quant_en,
        qsigned=qsigned,
        bits=qbits,

        inj_order=False,
    )

    # Golden data gathering
    accuracy, top1conf, top2diff, ave_loss = test_accuracy(goldeneye_model,
                                                           dataiter,
                                                           batchsize,
                                                           )
    return (accuracy, top1conf, top2diff, ave_loss)


def sweepFormat(num_format, bit_widths, qbit_widths, radices, model, dataset, batchsize, workers, ranges, float_ignore=8, verbose=False):
    format_count = 0
    data_sweep = {}
    explored = {}

    # expand all possible entries

    file_name = out_path + num_format + "_sweep.csv"
    with open(file_name, 'w') as fd:
        # header
        row_data = 'num, numformat, bitwidth, exp_bits, mantissa_bits, quant_en, qbits, accuracy, top1conf, top2diff, ave_loss\n'
        fd.write(row_data)

    for bit_width in tqdm(bit_widths):
        # severe lack of precision in floats below FLOAT_IGNORE. Skip these
        if "fp" in num_format and bit_width < float_ignore:
            continue

        # Sweep radix point
        for radix in range(1, bit_width, 2):
        # for radix in radices:
            exp_bits = bit_width - radix - 1  # also INT for fixed point
            mantissa_bits = bit_width - exp_bits - 1  # also FRAC for fixed point
            for quant_en in [False, True]:
            # for quant_en in [False]:
                if quant_en:
                    for qbits in qbit_widths:
                        if qbits > bit_width:
                            continue

                        run_goldeneye_profile(model, dataset, batchsize, workers,
                                              num_format, True, bit_width, radix, None,
                                              ranges, quant_en, True, qbits,
                                              verbose=verbose)
                        format_count += 1
                        data_sweep[format_count] = (num_format, bit_width, exp_bits, mantissa_bits, quant_en, qbits,
                                             accuracy, top1conf, top2diff, ave_loss)
                        # row_data = 'num, numformat, bitwidth, exp_bits, mantissa_bits, quant_en, qbits, accuracy, top1conf, top2diff, ave_loss\n'
                        row_data = "%d, %s, %d, %d, %d, %s, %d, %f, %f, %f, %f\n" %(format_count, num_format,
                                                                                    bit_width, exp_bits,
                                                                                    mantissa_bits, quant_en,
                                                                                    qbits, accuracy, top1conf,
                                                                                    top2diff, ave_loss)
                        with open(file_name, 'a+') as fd:
                            fd.write(row_data)
                else:  # no looping
                    (accuracy, top1conf, top2diff, ave_loss) = run_goldeneye_profile(model, dataset, batchsize, workers,
                                                                                     num_format, True, bit_width, radix, None,
                                                                                     ranges, quant_en, True, -1,
                                                                                     verbose=verbose
                                                                                     )

                    format_count += 1

                    row_data = "%d, %s, %d, %d, %d, %s, %d, %f, %f, %f, %f\n" % (format_count, num_format,
                                                                                 bit_width, exp_bits,
                                                                                 mantissa_bits, quant_en,
                                                                                 -1, accuracy, top1conf,
                                                                                 top2diff, ave_loss)
                    with open(file_name, 'a+') as fd:
                        fd.write(row_data)

                    data_sweep[format_count] = (num_format, bit_width, exp_bits, mantissa_bits, quant_en, -1,
                                         accuracy, top1conf, top2diff, ave_loss)

    output_name = num_format + "_sweep"
    df = save_data_df(out_path, output_name, data_sweep)

    if verbose:
        print("Count %s: %d" %(num_format, format_count))

    return format_count

def save_data_df(path, file_name, data):
    if not os.path.exists(path):
        os.makedirs(path)
    output = path + file_name
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.columns = ['num', 'numformat', 'bitwidth', 'exp_bits', 'mantissa_bits', 'quant_en', 'qbits',
                  'accuracy', 'top1conf', 'top2diff', 'ave_loss'
                                ]
    df.to_pickle(output + ".df")
    # df.to_csv(output + ".csv", index=False)

    return df


if __name__ == '__main__':

    # read in cmd line args
    check_args(sys.argv[1:])
    if getDebug(): printArgs()

    # common variables
    range_name = getDNN() + "_" + getDataset()
    range_path = getOutputDir() + "/networkRanges/" + range_name + "/"

    name = getDNN() + "_" + getDataset() + "_real" + getPrecision()
    out_path = getOutputDir() + "/numsys_sweep/" + name + "/"

    # get ranges
    ranges = load_file(range_path + "ranges_trainset_layer")

    # load data and model
    model = getNetwork(getDNN(), getDataset())
    model.eval()
    torch.no_grad()

    count = 0
    num_formats = ["fp_n", "fixedpt", "block_fp", "adaptive_fp"]

    # bit_widths = list(reversed(range(0, 33, 4)))
    # qbit_widths = list(reversed(range(0, 33, 4)))
    # radix_allowed = [1, 2, 4, 8, 16, 24, 31]

    bit_widths = [4, 8, 32]
    qbit_widths = [2, 8]
    radix_allowed = [1, 2, 4, 8, 16, 24, 31]


    # float sweep
    count += sweepFormat(num_formats[0], bit_widths, qbit_widths, radix_allowed,
                         model, getDataset(), getBatchsize(), getWorkers(), ranges,
                         verbose=getVerbose())

    # fxp sweep
    # count += sweepFormat(num_formats[1], bit_widths, qbit_widths,
    #                      model, getDataset(), getBatchsize(), getWorkers(), ranges,
    #                      verbose=getVerbose())

    # # block_fp sweep
    # bit_widths = list(reversed(range(8, 33, 4)))
    # count += sweepFormat(num_formats[2], bit_widths, qbit_widths,
    #                      model, getDataset(), getBatchsize(), getWorkers(), ranges,
    #                      verbose=getVerbose())
    #
    # # adaptiv_fp sweep
    # qbit_widths = list(reversed(range(0, 33, 4)))
    # count += sweepFormat(num_formats[3], bit_widths, qbit_widths,
    #                      model, getDataset(), getBatchsize(), getWorkers(), ranges,
    #                      verbose=getVerbose())

    print("Total Count:", count)