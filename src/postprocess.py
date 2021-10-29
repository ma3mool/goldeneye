from util import *
import concurrent.futures
import math
import pandas as pd
from tqdm import tqdm

MAX_INJ = 0
def setMax(value):
    global MAX_INJ
    MAX_INJ = value

def getMax():
    global MAX_INJ
    return MAX_INJ

def _layer_file_processing(data_in):
    inj_path, curr_layer, golden_data = data_in
    layerFileName = "layer" + str(curr_layer)
    layer_inj_data = load_file(inj_path + layerFileName)

    total_injections = 0
    _layer_mismatches = 0
    _layer_loss = 0
    SIZE_ = getMax()

    LayerData = []
    PRINT_INJ_INFO = True

    # NOT Optimal: should compute running average and running variance instead..
    # average
    for batch in range(len(layer_inj_data)):
        if total_injections >= SIZE_:
            break

        # if getQuantize_en() and getSingleBitFlip_en():
        #     batch_img_id, batch_inj_bit, batch_argmax, batch_inj_loss = fmap_inj_data[batch]
        # else:
        #     batch_img_id, batch_inj_H, batch_inj_W, batch_inj_val, batch_argmax, batch_inj_loss = fmap_inj_data[batch]
        batch_img_id, batch_argmax, batch_inj_loss = layer_inj_data[batch]

        for injection in range(len(batch_img_id)):
            if total_injections >= SIZE_:
                break

            total_injections += 1
            # injection info
            img_id = batch_img_id[injection]
            inj_loss = batch_inj_loss[injection]
            inj_label = batch_argmax[injection]

            # golden info
            gold_inf, gold_label, gold_conf, gold_top2diff, gold_loss = golden_data[img_id]

            # compare golden info with injection info
            if(inj_label != gold_label):
                _layer_mismatches += 1

            # compare loss
            _layer_loss += abs(gold_loss - inj_loss)

            injData = (img_id, gold_conf, gold_top2diff, gold_label, inj_label, gold_loss, inj_loss)
            LayerData.append(injData)

    ave_delta_loss = _layer_loss / total_injections

    # standard dev
    total_injections = 0
    var_sum = 0.0
    for batch in range(len(layer_inj_data)):
        if total_injections >= SIZE_:
            break

        # if getQuantize_en() and getSingleBitFlip_en():
        #     batch_img_id, batch_inj_bit, batch_argmax, batch_inj_loss = fmap_inj_data[batch]
        # else:
        #     batch_img_id, batch_inj_H, batch_inj_W, batch_inj_val, batch_argmax, batch_inj_loss = fmap_inj_data[batch]
        batch_img_id, batch_argmax, batch_inj_loss = layer_inj_data[batch]

        for injection in range(len(batch_img_id)):
            if total_injections >= SIZE_:
                break

            total_injections += 1
            # injection info
            img_id = batch_img_id[injection]
            inj_loss = batch_inj_loss[injection]
            inj_label = batch_argmax[injection]

            # golden info
            gold_inf, gold_label, gold_conf, gold_top2diff, gold_loss = golden_data[img_id]

            # compare loss to average_loss
            layer_for_var = abs(gold_loss - inj_loss) - ave_delta_loss
            var_sum += (layer_for_var * layer_for_var)

    var_delta_loss = var_sum / (total_injections-1)
    std_delta_loss = math.sqrt(var_delta_loss)


    if PRINT_INJ_INFO:
        fileName = inj_path + "layer" + str(curr_layer) + "_detailed"
        f_det = open(fileName + ".csv", "w+")

        outputString = "Injection, ImgID, GoldConf, GoldTop2Diff, GoldLabel, InjLabel, GoldLoss, InjLoss, DeltaLoss\n"
        f_det.write(outputString)

        count = 0
        for values in LayerData:
            img_id, gold_conf, gold_top2diff, gold_label, inj_label, gold_loss, inj_loss = values
            outputString = "%d, %d, %f, %f, %d, %d, %f, %f, %f\n" %(count,
                                                                    img_id,
                                                                    gold_conf,
                                                                    gold_top2diff,
                                                                    gold_label,
                                                                    inj_label,
                                                                    gold_loss,
                                                                    inj_loss,
                                                                    abs(gold_loss - inj_loss),
                                                                   )
            f_det.write(outputString)
            count += 1
        f_det.close()

        # create a data frame
        df = pd.read_csv(fileName + ".csv")
        df.to_pickle(fileName + ".pkl")


    return curr_layer, _layer_mismatches, ave_delta_loss, std_delta_loss

if __name__ == "__main__":
    # Read in cmd line args
    check_args(sys.argv[1:])
    if getDebug(): printArgs()

    # PATHS
    range_name = getDNN() + "_" + getDataset()
    range_path = getOutputDir() + "/networkRanges/" + range_name + "/"

    name = getDNN() + "_" + getDataset() + "_real" + getPrecision() + "_sim" + getFormat()
    if getQuantize_en(): name += "_" + "quant"

    range_path = getOutputDir() + "/networkRanges/" + range_name+ "/"
    profile_path= getOutputDir() + "/networkProfiles/" + name + "/"
    data_susbet_path = getOutputDir() + "/data_subset/" + name + "/"
    inj_path = getOutputDir() + "/injections/" + name + "/"
    out_path = getOutputDir() + "/postprocess/" + name + "/"

    if getTraining_en():
        inj_path += "training/"
        out_path += "training/"
    else: #testing
        inj_path += "testing/"
        out_path += "testing/"

    # load important info: ranges, mapping, good images
    ranges = load_file(range_path + "ranges_trainset_layer")
    golden_data = load_file(profile_path + "golden_data")

    LAYERS = len(ranges)
    INJ = getInjections()
    setMax(INJ)

    # Data Processing
    processing_tuple = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=getWorkers()) as executor:
        # prep parallel script data
        for layer in tqdm(range(LAYERS)):
            processing_tuple.append((inj_path, layer, golden_data))


        # launch in parallel
        results = executor.map(_layer_file_processing, processing_tuple)

    print("FINISHED")

    # split up for write to disk
    layer_mismatches = []
    layer_loss = []
    layer_loss_std = []
    layer_counter = 0
    for i in results:
        layer_i, mismatches_i, loss_i, std_loss_i = i
        assert(layer_i == layer_counter)
        layer_mismatches.append([layer_i, mismatches_i])
        layer_loss.append([layer_i, loss_i])
        layer_loss_std.append([layer_i, std_loss_i])

        layer_counter += 1

    # save results
    save_data(out_path, "data_mismatches_" + str(INJ), layer_mismatches)
    save_data(out_path, "data_loss_" + str(INJ), layer_loss)

    f = open(out_path + "mismatches_" + str(INJ) + ".csv", "w+")
    for i in range(len(layer_mismatches)):
        outputString = "%d, %d\n" %(i, layer_mismatches[i][1])
        f.write(outputString)
    f.close()

    f = open(out_path + "loss_" + str(INJ) + ".csv", "w+")
    for i in range(len(layer_loss)):
        outputString = "%d, %f\n" %(i, layer_loss[i][1])
        f.write(outputString)
    f.close()

    f = open(out_path + "loss_and_mismatches_" + str(INJ) + ".csv", "w+")
    outputString = "fmap_id, Mismatches, Loss, Sample_Size, STD_Loss, Error_Loss, Proportion, error_mismatches_proportion,error_mismatches\n"
    f.write(outputString)
    for i in range(len(layer_loss)):
        z = 2.576 # Z value for 99% confidence
        error_loss_i = (z * layer_loss_std[i][1]) / math.sqrt(INJ)
        mismatch_rate_i = 1.0 * layer_mismatches[i][1] / INJ
        error_mismatch_i = z * math.sqrt(mismatch_rate_i * (1.0 - mismatch_rate_i)/ INJ)
        error_mismatch_bars = error_mismatch_i * INJ

        outputString = "%d, %d, %f, %d, %f, %f, %f, %f, %f\n" %(i, layer_mismatches[i][1], layer_loss[i][1], INJ, layer_loss_std[i][1], error_loss_i, mismatch_rate_i, error_mismatch_i, error_mismatch_bars)
        f.write(outputString)
    f.close()
