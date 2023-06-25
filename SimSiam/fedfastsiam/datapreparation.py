import json
from PIL import Image
import os, glob
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import pickle

RESCALE_SIZE = (120, 100)


class ZenseactSSLDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.roadcondition = None

    def __getitem__(self, idx):
        dat = self.data[idx]
        if self.transform:
            dat = self.transform(dat)
        dummy_label = 0
        return dat, dummy_label

    def __len__(self):
        return len(self.data)
    

class ZenseactMetadata(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        meta_dat = self.data[idx]
        return meta_dat

    def __len__(self):
        return len(self.data)
    

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
        return [q, views]
    

def generate_ssl_data(size=25_000):
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


def generate_meta_data(size=25_000):
    parent_directory = "../../../mnt/nfs_mount/single_frames"

    # Find all folders with a 6-digit name
    folder_pattern = os.path.join(parent_directory, "[0-9]" * 6)

    # Get the list of matching folders
    folders = glob.glob(folder_pattern)
    folders = folders[0:size] # *size* folders

    meta_data = []
    for folder in folders:
        id = os.path.basename(folder) # id = foldername
        # metainformation
        metadata = f"../../../mnt/nfs_mount/single_frames/{id}/metadata.json"
        f = open(metadata)
        metadata = json.load(f)
        weather_condition = metadata["scraped_weather"]
        meta_data.append(weather_condition)

    return meta_data


def load_data_noniid(zenseactssldataset, zenseactmetadata, num_clients, batch_size):
    num_data_per_client = len(zenseactssldataset) // num_clients
    
    client_indices = {}
    for i in range(num_clients):
        client_indices[str(i+1)] = []

    def add_index(client_num, index):
        if client_num > num_clients:
            return
        elif len(client_indices[str(client_num)]) >= num_data_per_client:
            add_index(client_num+1, index)
        else:
            client_indices[str(client_num)].append(index)

    for i in range(len(zenseactmetadata)):
        weather = zenseactmetadata[i]
        if weather == "partly-cloudy-day":
            add_index(1, i)
        elif weather == "cloudy":
            add_index(2, i)
        elif weather == "clear-day":
            add_index(3, i)
        elif weather == "rain":
            add_index(4, i)
        elif weather == "clear-night":
            add_index(5, i)
        elif weather == "snow":
            add_index(6, i)
        elif weather == "partly-cloudy-night":
            add_index(7, i)
        elif weather == "fog":
            add_index(8, i)
        else:
            add_index(9, i)

    local_dataloaders = []
    for client, indices in client_indices.items():
        local_datasets = Subset(zenseactssldataset, indices)
        local_dataloaders.append(DataLoader(local_datasets, batch_size=batch_size))

    return local_dataloaders
    
    

def load_data_iid(trainset, num_clients, batch_size):
    shuffled_indices = torch.randperm(len(trainset))
    # Get the total number of indices
    num_indices = len(shuffled_indices)

    # Determine the number of indices per client
    indices_per_client = num_indices // num_clients

    # Distribute indices among clients
    local_dataloaders = []
    for i in range(num_clients):
        start = i * indices_per_client
        if i == num_clients - 1:  # Last client gets remaining indices
            end = num_indices
        else:
            end = start + indices_per_client
        client_indices = shuffled_indices[start:end]

        # create subset of dataset
        subset = Subset(trainset, client_indices)
        # create dataloader 
        local_dataloder = torch.utils.data.DataLoader(subset, batch_size=batch_size)

        local_dataloaders.append(local_dataloder)
    return local_dataloaders


def create_datasets(num_clients, dataset_size=25_000, batch_size=32, num_views=4):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    augmentation = [
        transforms.RandomResizedCrop(RESCALE_SIZE, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    if os.path.exists("ssl_data.pkl"):
        print("data file found")
        with open("ssl_data.pkl", "rb") as file:
            data = pickle.load(file)
    else:
        print("generating data. This might take a while.")
        data = generate_ssl_data(dataset_size)
        with open("ssl_data.pkl", "wb") as file:
            pickle.dump(data, file)

    if os.path.exists("meta_data.pkl"):
        print("data file found")
        with open("meta_data.pkl", "rb") as file:
            meta_data = pickle.load(file)
    else:
        print("generating meta data. This might take a while.")
        meta_data = generate_meta_data(dataset_size)
        with open("meta_data.pkl", "wb") as file:
            pickle.dump(meta_data, file)

    zenseactmetadata = ZenseactMetadata(meta_data)
    zenseactssldataset = ZenseactSSLDataset(data, transform=TwoCropsTransform(transforms.Compose(augmentation), num_views))

    local_dataloaders = load_data_noniid(zenseactssldataset, zenseactmetadata, num_clients, batch_size)

    return local_dataloaders