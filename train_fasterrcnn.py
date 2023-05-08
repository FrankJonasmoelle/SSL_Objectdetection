import torch
import torchvision
import pandas as pd
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class ZenseactLabeledDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.df = pd.read_csv("df_bounding_boxes.csv")
        # image tensor

        # box coordinates

        # label (1)

        # id


        self.data = data
        self.n_samples = len(data)

    def __getitem__(self, index):
        data = self.df.iloc[index]

        id = data['image_id']
        # TODO: could be several rows
        x_min, y_min, x_max, y_max = data['x_min'], data['y_min'], data['x_max'], data['y_max']

        img = self.data[index]["image_id"]
        boxes = self.data[index]["boxes"]
        labels = self.data[index]["labels"]
        id = self.data[index]["image_id"]
        target = {}
        target['boxes'] = [x_min, y_min, x_max, y_max]
        target['labels'] = labels
        return img, target, id
    
    def __len__(self):
        return self.n_samples
    


    # def __getitem__(self, index):
    #     img = self.data[index]["data"]
    #     boxes = self.data[index]["boxes"]
    #     labels = self.data[index]["labels"]
    #     id = self.data[index]["image_id"]
    #     target = {}
    #     target['boxes'] = boxes
    #     target['labels'] = labels
    #     return img, target, id
    
    # def __len__(self):
    #     return self.n_samples
    

def create_fasterrcnn(modelpath):
    pretrained_weights = torch.load(modelpath, map_location=torch.device('cuda')) # cpu

    resnet_model = resnet50()

    # Load the pre-trained weights into the model
    resnet_model.load_state_dict(pretrained_weights)

    # Remove the last fully connected layer to use the model as a backbone
    backbone = torch.nn.Sequential(*(list(resnet_model.children())[:-2]))

    # Add a Feature Pyramid Network (FPN) on top of the backbone
    backbone = resnet_fpn_backbone(backbone, pretrained=False)

    # Create the anchor generator
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # Set the number of classes
    num_classes = 2  # Vehicles + 1 background class

    # Create the Faster R-CNN model
    model = FasterRCNN(backbone,
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator)
    return model

if __name__=="__main__":
    modelpath = "model.pth"
    model = create_fasterrcnn(modelpath)

