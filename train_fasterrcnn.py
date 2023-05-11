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


from SimSiam.simsiam.fastsiam import *


RESCALE_SIZE = (120, 100)

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
    
def generate_data(df, index):
    data = df[df["id"] == index]

    num_boxes = len(data)

    box_coordinates = []
    for i in range(num_boxes):
        box_coordinates.append(torch.tensor(data.iloc[i][["x_min", "y_min", "x_max", "y_max"]].astype("float").to_numpy()))

    if num_boxes > 1:
        box_coordinates = torch.stack(box_coordinates, axis=0)
    elif num_boxes == 1:
        box_coordinates = box_coordinates[0]
        box_coordinates = box_coordinates.view(1,4)
    else:
        pass

    labels = torch.ones(num_boxes, dtype=torch.int64)

    image_id = data.iloc[0]["image_id"]
    image_path = f"../../../mnt/nfs_mount/single_frames/{image_id}/camera_front_blur/"
    image_path = glob.glob(image_path + "*.jpg")
    image = Image.open(image_path[0]).convert('RGB')

    downsampled_image = image.resize(RESCALE_SIZE)
    transform = transforms.Compose([
                transforms.ToTensor()
            ])

    downsampled_image = transform(downsampled_image)  # Apply the transform

    # stack it to dict
    image_dict = {}
    image_dict["data"] = downsampled_image
    image_dict["boxes"] = box_coordinates
    image_dict["labels"] = labels
    image_dict["image_id"] = index
    return image_dict


def generate_all_data():
    df = pd.read_csv("df_bounding_boxes.csv", dtype=str)
    df["id"] = df["image_id"].astype(int)
    df = df[["id", "image_id", "x_min", "y_min", "x_max", "y_max"]]
    unique_ids = df["id"].unique()     # all ids / indices

    data = []
    for index in unique_ids:
        try:
            data_dict = generate_data(df, index)
            data.append(data_dict)
        except:
            pass

    return data


def custom_resnet_fpn_backbone(resnet, return_layers, in_channels_stage2=256, out_channels=256):
    backbone = resnet
    # Remove the fully connected layer (fc)
    del backbone.fc

    # Extract the layers from the ResNet model
    return_layers = return_layers
    in_channels_list = [in_channels_stage2, in_channels_stage2 * 2, in_channels_stage2 * 4, in_channels_stage2 * 8]
    out_channels = out_channels

    # Create the FPN using the extracted layers
    fpn = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    return fpn


def create_fasterrcnn(modelpath):
    if modelpath:
        resnet_model = FastSiam()
        pretrained_weights = torch.load(modelpath, map_location=torch.device('cuda')) # cpu
        resnet_model.load_state_dict(pretrained_weights)
        resnet_model = resnet_model.backbone
    else:
        resnet_model = resnet50()

    # Define which layers from the ResNet model to use as output for the FPN
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    # Create the FPN on top of your randomly-initialized ResNet-50 model
    custom_backbone = custom_resnet_fpn_backbone(resnet_model, return_layers)

    # Define the number of classes for your detection task (including background class)
    num_classes = 2  # For example, 91 classes including background for the COCO dataset

    # Create the Faster R-CNN model with the custom backbone
    model = FasterRCNN(backbone=custom_backbone, num_classes=num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__=="__main__":
    """
    python3 train_fasterrcnn.py
    """
    torch.cuda.empty_cache()

    batch_size = 16
    modelpath = "fastsiam.pth"
    model = create_fasterrcnn(modelpath)

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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 20

    scaler = GradScaler()

    loss_per_epoch = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets, id in trainloader:
            torch.cuda.empty_cache()
            try:
                print("trying to put it to the gpu")
                optimizer.zero_grad()

                with autocast():
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #[{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)
                    # print(loss_dict)
                    losses = sum(loss for loss in loss_dict.values())
                    epoch_loss += losses.item()

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

              #  losses.backward()
              #  optimizer.step()
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
        try:
            print(f"loss for epoch {epoch}: {epoch_loss / len(trainloader)}")
            loss_per_epoch.append(epoch_loss / len(trainloader))
        except Exception as e:
            print(e)

    PATH = "faster_rcnn_fastsiam.pth"
    torch.save(model.state_dict(), PATH)