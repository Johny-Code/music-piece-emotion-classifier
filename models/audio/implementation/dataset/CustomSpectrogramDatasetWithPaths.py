import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class CustomSpectrogramDatasetWithPaths(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.transform = transform
        self.samples = []
        self.target_dict = {'happy': [1., 0., 0., 0.],
                            'angry': [0., 1., 0., 0.],
                            'sad': [0., 0., 1., 0.],
                            'relaxed': [0., 0., 0., 1.]}

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                label = self.target_dict[class_name]
                self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label), img_path
    
    def get_sample_and_path(self, idx):
        return self.samples[idx]
    