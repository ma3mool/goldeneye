import os
import cv2
from util import *
from PIL import Image, ImageDraw
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from goldeneye import *
from torchvision.transforms.functional import to_pil_image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2

def check_labels_tensors(tensor1, tensor2):
    # Convert tensors to sets
    set1 = set(tensor1.numpy())
    set2 = set(tensor2.numpy())

    # Find the intersection of sets (matched elements)
    intersection = set1.intersection(set2)

    # Count the number of matched elements (including duplicates)
    ground_truth_boxes = sum(min(tensor1.numpy().tolist().count(element),
                                 tensor2.numpy().tolist().count(element))
                             for element in intersection)

    # Count the number of unmatched elements
    wrong_boxes = abs(len(tensor2) - ground_truth_boxes)
    return (ground_truth_boxes, wrong_boxes)


def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)):
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']):
            if score > threshold:
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

def calculate_iou(box1, box2):
    # Calculate coordinates of intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Calculate area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Calculate areas of the boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def calculate_iou_for_all_boxes(gt_boxes, pred_boxes, threshold=0.50):
    iou_scores = []
    for box1 in gt_boxes:
        iou_scores_row = []
        for box2 in pred_boxes:
            iou_score = calculate_iou(box1, box2)
            if iou_score >= threshold:
                iou_scores_row.append(iou_score)
        if iou_scores_row:
            iou_scores.append(iou_scores_row)
    return iou_scores

def calculate_iou_p(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xA = torch.max(x1, x1_)
    yA = torch.max(y1, y1_)
    xB = torch.min(x2, x2_)
    yB = torch.min(y2, y2_)
    inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def calculate_precision_recall_f1(predicted_boxes, true_boxes, iou_threshold):
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(predicted_boxes.size(0)):
        true_positives = 0
        false_positives = 0
        total_true_boxes = true_boxes.size(0)
        total_predicted_boxes = predicted_boxes.size(0)

        if total_true_boxes == 0 or total_predicted_boxes == 0:
            # Handle cases where there are no true boxes or predicted boxes
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
        else:
            for j in range(total_predicted_boxes):
                matched = False
                for k in range(total_true_boxes):
                    iou = calculate_iou(predicted_boxes[j], true_boxes[k])
                    if torch.max(iou) >= iou_threshold:
                        true_positives += 1
                        matched = True
                        break

                if not matched:
                    false_positives += 1

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / total_true_boxes
            f1_score = 2 * (precision * recall) / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return precisions, recalls, f1_scores
@torch.no_grad()
def gather_golden(goldeneye, data_iter, cuda_en=True, precision='FP32', verbose=False, debug=False,out_saved_imgs_path=""):
    golden_data = {}
    good_boxes = 0
    bad_boxes = 0

    criterion = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).cuda()
    criterion.train()

    for images, labels, index in tqdm(data_iter):
        inf_model = goldeneye.declare_neuron_fi(function=goldeneye.apply_goldeneye_transformation)
        if precision == 'FP16':
            images = images.half()

        images = list(img.cuda() for img in images)
        labels = [{k: v.cuda() for k, v in t.items()} for t in labels]

        with torch.no_grad():
            # with torch.autocast(device_type='cuda', dtype=torch.float16# ):
            out_pred = make_prediction(inf_model, images, 0.35)
            out_loss = criterion(images, labels)

            cls_loss = out_loss['loss_classifier'].item()
            bbox_loss = out_loss['loss_box_reg'].item()
            obj_loss = out_loss['loss_objectness'].item()
            rpn_loss = out_loss['loss_rpn_box_reg'].item()

        for idx, img in enumerate(images):
            try:
                img = to_pil_image(img)
                draw = ImageDraw.Draw(img)
                true_box_corrds = labels[idx]['boxes'].int().cpu()
                pred_box_corrds = out_pred[idx]['boxes'].int().cpu()
                pred_labels = out_pred[idx]['labels'].cpu()
                gr_label = labels[idx]['labels'].cpu()
                pred_box_conf = out_pred[idx]['scores']

                precisions, recalls, f1_scores = calculate_precision_recall_f1(out_pred[idx]['boxes'], labels[idx]['boxes'],0.60)

                for box, pd_label, score, in zip(pred_box_corrds, pred_labels, pred_box_conf):
                    if score > 0.4:
                        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="blue", width=4)
                        draw.text((box[0], box[1]), f"{pd_label}: {score:.2f}", fill="red")
                        img.save(f"{out_saved_imgs_path}/output_{index[idx]}.png")

                # for i in range(len(precisions)):
                #     print(
                #         f"Image {i + 1} - Precision: {precisions[i]:.2f}, Recall: {recalls[i]:.2f}, F1 Score: {f1_scores[i]:.2f}")
            except:
                continue

            correct_boxes = check_labels_tensors(gr_label.cpu(), pred_labels.cpu())[0]
            good_boxes += correct_boxes

            wrong_boxes = check_labels_tensors(gr_label.cpu(), pred_labels.cpu())[1]
            bad_boxes += wrong_boxes

            gr_boxes_list = labels[idx]['boxes'].int().cpu().tolist()
            pred_boxes_list = out_pred[idx]['boxes'].int().cpu().tolist()

            gr_boxes = len(gr_label)
            pred_boxes = len(pred_labels)

            bbox_iou = calculate_iou_for_all_boxes(gr_boxes_list, pred_boxes_list)
            bbox_ious = []

            for b_iou in bbox_iou:
                for c_iou in b_iou:
                    if c_iou == 0.0:
                        continue
                    bbox_ious.append(c_iou)
            """img_id (list), true_box_corrds(list), pred_box_corrds (list), gr_labels [ ] , pred_labels [ ] , bbox_ious, gr_boxes_count (int), pred_boxes_count (int), precisions, recall,  correct_box_count (int), wrong_box_count (int), box_conf (float), cls_loss (float), bbox_loss (float), obj_loss (float), rpn_loss (float)."""
            img_id = index[idx]
            img_tuple = (true_box_corrds.tolist(),pred_box_corrds.tolist(), gr_label.tolist(), pred_labels.tolist(), bbox_iou, pred_box_conf.tolist(), precisions, recalls, f1_scores , gr_boxes, pred_boxes,correct_boxes, wrong_boxes, cls_loss, bbox_loss, obj_loss, rpn_loss)
            assert (img_id not in golden_data)  # we shouldn't have duplicates in the golden data
            golden_data[img_id] = img_tuple
        torch.cuda.empty_cache()

    total_boxes = good_boxes + bad_boxes
    return golden_data, good_boxes, bad_boxes, total_boxes

"""img_id (int), gr_labels [ ] , pred_labels [ ] , gr_boxes_count (int), pred_boxes_count (int),precisions, recalls, f1_scores ,correct_box_count (int), wrong_box_count (int), box_conf (float), cls_loss (float), bbox_loss (float), obj_loss (float), rpn_loss (float)."""

def save_data_df(path, file_name, data):
    if not os.path.exists(path):
        os.makedirs(path)

    output = path + file_name
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.columns = ['img_id', 'true_box_coords', 'pred_box_coords', 'gr_labels', 'pred_labels', 'bbox_ious', 'pred_box_confs', 'precisions', 'recalls', 'f1-scores','gr_boxes',
                  'pred_boxes','correct_boxes', 'wrong_boxes', 'cls_loss', 'bbox_loss', 'obj_loss','rpn_loss']
    df.to_pickle(output + ".df")
    df.to_csv(output + ".csv", index=False)
    return df

if __name__ == '__main__':

    # read in cmd line args
    check_args(sys.argv[1:])
    if getDebug():
        printArgs()

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
    imgs_out_folder = out_path+ "/output_images/"

    if os.path.exists(imgs_out_folder):
        pass
    else:
        os.mkdir(imgs_out_folder)
    # get ranges
    ranges = load_file(range_path + "ranges_trainset_layer")

    # load data and model
    dataiter = load_dataset(getDataset(), getBatchsize(), workers=getWorkers())
    model = getNetwork(getDNN(), getDataset())
    print(" Dataset & Model loaded....")
    model.eval()
    torch.no_grad()

    exp_bits = getBitwidth() - getRadix() - 1  # also INT for fixed point
    mantissa_bits = getRadix()  # getBitwidth() - exp_bits - 1  # also FRAC for fixed point

    # no injections during profiling
    assert (getInjections() == -1)
    assert (getInjectionsLocation() == 0)

    # init PyTorchFI
    baseC = 3
    if "IMAGENET" in getDataset():
        baseH = 224
        baseW = 224
    elif "CIFAR" in getDataset():
        baseH = 32
        baseW = 32

    elif "COCO" in getDataset():
        baseH = 480
        baseW = 640

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
        layer_max=ranges,
        bits=getBitwidth(),
        qsigned=True,

        inj_order=getInjectionsLocation(),
    )

    # Golden data gathering
    golden_data, good_boxes, bad_boxes, total_boxes = gather_golden(goldeneye_model, dataiter,
                                                                    getBatchsize(), precision=getPrecision(),
                                                                    verbose=getVerbose(), debug=getDebug(), out_saved_imgs_path=imgs_out_folder)
    # Golden data gathering
    output_name = "golden_data"
    save_data(out_path, output_name, golden_data)
    df = save_data_df(out_path, output_name, golden_data)

    # Print Summary Statistics
    summaryDetails = ""
    summaryDetails += "===========================================\n"
    summaryDetails += "%s\n" % (name)
    summaryDetails += "Accuracy based on Pred_True Boxes: \t%0.2f%%\n" % (good_boxes / total_boxes * 100.0)
    # summaryDetails += "Ave Conf: \t%0.2f%%\n" % (df["box_conf"].mean())
    summaryDetails += "===========================================\n"

    # save stats
    stats_file = open(out_path + "stats.txt", "w+")
    n = stats_file.write(summaryDetails)
    stats_file.close()

    if getVerbose():
        print(summaryDetails)
        print("===========================================")
        print(name)
        print("Accuracy based on Pred_True Boxes: \t%0.2f%%" % (good_boxes / total_boxes * 100.0))
        # print("Ave Conf: \t%0.2f%%" % (df["box_conf"].mean()))
        print("===========================================")
