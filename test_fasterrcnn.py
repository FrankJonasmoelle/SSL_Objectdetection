import torch
import torchvision
from PIL import Image
import pandas as pd
import pickle
import glob
import os
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from matplotlib.patches import Rectangle
import cv2


from SimSiam.simsiam.fastsiam import *
from train_fasterrcnn import *


class ZenseactLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        img = self.data[index]["data"]
        boxes = self.data[index]["boxes"]
        labels = self.data[index]["labels"]
        id = self.data[index]["image_id"]
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        return img, target, id
    
    def __len__(self):
        return self.n_samples


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1a, y1a, x2a, y2a = box2

    xi1, yi1, xi2, yi2 = max(x1, x1a), max(y1, y1a), min(x2, x2a), min(y2, y2a)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = (x2 - x1) * (y2 - y1) + (x2a - x1a) * (y2a - y1a) - intersection

    return intersection / union



if __name__=="__main__":
    batch_size = 16

    # FastSiam
    # model = create_fasterrcnn("fastsiam.pth")
    # weights = torch.load("faster_rcnn_fastsiam.pth", map_location=torch.device("cuda"))

    # resnet50
    model = create_fasterrcnn(None)
    weights = torch.load("faster_rcnn_resnet50.pth", map_location=torch.device("cuda"))

    model.load_state_dict(weights)

    # dataset preparation 
    if os.path.exists("all_data.pkl"):
        with open("all_data.pkl", "rb") as file:
            all_data = pickle.load(file)
    else:
        print("generating data. This might take a while.")
        all_data = generate_all_data()
        with open("all_data.pkl", "wb") as file:
            pickle.dump(all_data, file)

    dataset = ZenseactLabeledDataset(all_data)

    del all_data
    torch.cuda.empty_cache()

    # train-test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(0)
    trainset, testset = random_split(dataset, [train_size, test_size])

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    results=[]
    detection_threshold = 0.1 # the lower, the less we keep
    iou_threshold = 0.5

    model.to(device)
    model.eval()

    TP = 0
    FP = 0
    FN = 0
    iou_threshold = 0.5
    detection_threshold = 0.1

    for images, targets, id in testloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)

        for i in range(len(targets)):
            gt_boxes = targets[i]["boxes"].tolist()
            pred_boxes = predictions[i]["boxes"].tolist()
            pred_scores = predictions[i]["scores"].tolist()

            # Apply NMS to filter bounding boxes
            try:
                selected_indices = torchvision.ops.nms(torch.tensor(pred_boxes), torch.tensor(pred_scores), detection_threshold)
                pred_boxes = [pred_boxes[i] for i in selected_indices]
                pred_scores = [pred_scores[i] for i in selected_indices]
            except Exception as e:
                print(e)
                continue

            # For each ground truth box, find the prediction with the highest IoU
            for gt_box in gt_boxes:
                max_iou = -1
                max_iou_index = -1
                for idx, pred_box in enumerate(pred_boxes):
                    current_iou = iou(gt_box, pred_box)
                    if current_iou > max_iou:
                        max_iou = current_iou
                        max_iou_index = idx

                if max_iou >= iou_threshold:
                    TP += 1
                    # Remove the matched prediction box to avoid double counting
                    pred_boxes.pop(max_iou_index)
                    pred_scores.pop(max_iou_index)
                else:
                    FN += 1

            # The remaining prediction boxes are considered as false positives
            FP += len(pred_boxes)

    # Calculate Precision, Recall, and Average Precision (AP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    AP = precision * recall / (precision + recall) * 2

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Average Precision (AP): {AP}")
