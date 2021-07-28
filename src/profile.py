from util import *
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# goes through dataset and finds correctly classified images, and corresponding data
@torch.no_grad()
def gather_golden(model, data_iter, cuda_en=True, precision='FP16', verbose=False, debug=False):
    golden_data = {}
    processed_elements = 0
    good_imgs = 0
    bad_imgs = 0

    criterion = nn.CrossEntropyLoss(reduction='none')
    counter = 0
    for input_data in tqdm(data_iter):
        # if debug:
        #     if processed_elements >= max_elements:
        #         break

        # prepare the next batch for inference
        images, labels, img_ids, index = input_data
        if precision == 'FP16': images = images.half()
        if cuda_en:
            images = images.cuda()
            labels = labels.cuda()

        output = model(images) # run an inference
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
            img_tuple = (correct_inf, inf_label, inf_conf, inf_top2diff, inf_loss)
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

    ### common variables
    name = getDNN() + "_" + getDataset() + "_" + getPrecision()
    range_path = getOutputDir() + "/networkRanges/" + name + "/"
    out_path = getOutputDir() + "/networkProfiles/" + name + "/"
    subset_path = getOutputDir() + "/data_subset/" + name + "/"

    ### load data and model
    dataiter = load_dataset(getDataset(), getBatchsize(), workers = getWorkers())
    model = getNetwork(getDNN(), getDataset())
    model.eval()
    torch.no_grad()

    ### Golden data gathering
    golden_data, good_imgs, bad_imgs, total_imgs = gather_golden(model, dataiter, \
            getBatchsize(), precision=getPrecision(), verbose=getVerbose(), debug=getDebug())

    ### Golden data gathering
    output_name = "golden_data"
    save_data(out_path, output_name, golden_data)
    df = save_data_df(out_path, output_name, golden_data)

    ### Print Summary Statistics
    if getVerbose():
        print("===========================================")
        print(name)
        print("Accuracy: \t%0.2f%%" %(good_imgs / total_imgs * 100.0))
        print("Ave Loss: \t%0.2f" %(df["inf_loss"].mean()))
        print("Ave Conf: \t%0.2f%%" %(df["inf_conf"].mean()))
        print("Ave Top2Diff: \t%0.2f%%" %(df["inf_top2diff"].mean()))
        print("===========================================")
