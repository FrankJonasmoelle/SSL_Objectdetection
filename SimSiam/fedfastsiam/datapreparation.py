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


def load_data_iid(trainset, num_clients, batch_size):
    shuffled_indices = torch.randperm(len(trainset))
    training_x = trainset.data[shuffled_indices]
    training_y = torch.Tensor(trainset.targets)[shuffled_indices]

    split_size = len(trainset) // num_clients
    split_datasets = list(
                        zip(
                            torch.split(torch.Tensor(training_x), split_size),
                            torch.split(torch.Tensor(training_y), split_size)
                        )
                    )
    new_split_datasets = [(dataset[0].numpy(), dataset[1].tolist()) for dataset in split_datasets]
    new_split_datasets = [(dataset[0], list(map(int, dataset[1]))) for dataset in new_split_datasets]

    local_trainset = [ZenseactDataset(local_dataset[0], local_dataset[1], is_train=True) for local_dataset in new_split_datasets]

    local_dataloaders = [DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) for dataset in local_trainset]
    return local_dataloaders


def create_datasets(num_clients, dataset_size=5_000, batch_size=64, num_views=4):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""

    
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
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

    local_dataloaders = load_data_iid(trainset, num_clients, batch_size)

    return local_dataloaders