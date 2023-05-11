import json
from PIL import Image
import os, glob
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms


RESCALE_SIZE = (120, 100)

class ZenseactDataset(Dataset):
    def __init__(self, size=50_000, transform=None):
        self.image_paths = []
        self.transform = transform

        # The parent directory containing the folders
        parent_directory = "../../../mnt/nfs_mount/single_frames"

        # Find all folders with a 6-digit name
        folder_pattern = os.path.join(parent_directory, "[0-9]" * 6)

        # Get the list of matching folders
        folders = glob.glob(folder_pattern)
        self.folders = folders[:size] # first 5000 folders

        for folder in self.folders:
            id = os.path.basename(folder) # id = foldername
            # load image
            image_path = f"../../../mnt/nfs_mount/single_frames/{id}/camera_front_blur/"
            image_path = glob.glob(image_path + "*.jpg")
            self.image_paths.append(image_path)


    def __len__(self):
        return len(self.folders)


    def __getitem__(self, idx):
        image_path = self.image_paths[idx][0]
        # load image
        image = Image.open(image_path).convert('RGB')
        # resize image
        downsampled_image = image.resize(RESCALE_SIZE)

        if self.transform:
            downsampled_image = self.transform(downsampled_image)

        dummy_label = 0        
        return downsampled_image, dummy_label
    

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, num_views):
        self.base_transform = base_transform
        self.num_views = num_views

    def __call__(self, x):
        q = [self.base_transform(x)]
        views = []
        for view in range(self.num_views):
            view = self.base_transform(x)
            views.append(view)
        # q = q.extend(views)
        return [q, views]


def prepare_data(dataset_size=5_000, batch_size=64, num_views=4):
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
    trainset = ZenseactDataset(dataset_size, transform=TwoCropsTransform(transforms.Compose(augmentation), num_views))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainloader