import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import sys

#https://pytorch.org/vision/main/datasets.html
class LightDataset(Dataset):
    def __init__(self, rootpath, transform=None):
        self.transform = transform
        self.rootpath = rootpath
        self.data_dir = os.path.join(rootpath, "discriminator_data")

        # Gather all discriminator image paths
        self.image_paths = list()
        for filename in os.listdir(self.data_dir):
            self.image_paths.append(os.path.join(self.data_dir, filename))

    def __getitem__(self, index):
        image_path = self.image_paths[index % len(self.image_paths)]
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.image_paths)

class DarkDataset(Dataset):
    def __init__(self, rootpath, transform=None):
        self.transform = transform
        self.rootpath = rootpath
        self.data_dir = os.path.join(rootpath, "generator_data")

        # Gather all generator image paths
        self.image_paths = list()
        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            for filename in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, filename))

    def __getitem__(self, index):
        image_path = self.image_paths[index % len(self.image_paths)]
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.image_paths)