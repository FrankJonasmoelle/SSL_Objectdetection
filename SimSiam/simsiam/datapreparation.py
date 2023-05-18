import json
from PIL import Image
import os, glob
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms


RESCALE_SIZE = (120, 100)


class ZenseactSSLDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        dat = self.data[idx]
        if self.transform:
            dat = self.transform(dat)
        dummy_label = 0
        return dat, dummy_label

    def __len__(self):
        return len(self.data)
    

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


def generate_ssl_data(size=50_000):
    parent_directory = "../../../mnt/nfs_mount/single_frames"

    # Find all folders with a 6-digit name
    folder_pattern = os.path.join(parent_directory, "[0-9]" * 6)

    # Get the list of matching folders
    folders = glob.glob(folder_pattern)
    folders = folders[0:size] # *size* folders

    image_data = []
    for folder in folders:
        id = os.path.basename(folder) # id = foldername
        # load image
        image_path = f"../../../mnt/nfs_mount/single_frames/{id}/camera_front_blur/"
        image_path = glob.glob(image_path + "*.jpg")
        image = Image.open(image_path[0]).convert('RGB')
        # resize image
        downsampled_image = image.resize(RESCALE_SIZE)

        image_data.append(downsampled_image)
    return image_data


def prepare_data(data, batch_size=32):
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(RESCALE_SIZE, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize
    ]
    #train_set, val_set = torch.utils.data.random_split(trainset, [45000, 5000])
    trainset = ZenseactSSLDataset(data, transform=TwoCropsTransform(transforms.Compose(augmentation)))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainloader