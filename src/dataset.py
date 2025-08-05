import os

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class RetinaDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transforms=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask > 0] = 1.0

        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']
        
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.float()
        
        return image_tensor, mask_tensor