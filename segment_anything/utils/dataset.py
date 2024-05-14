import os
import numpy as np 
from PIL import Image
from typing import Type

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.data_images = os.listdir(image_dir)
        self.mask_images = os.listdir(mask_dir)
        
        assert len(self.data_images) == len(self.mask_images), "<Error> The number of images and the number of masks must be the same."

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data_images[idx])
        mask_path = os.path.join(self.mask_dir, self.data_images[idx])  

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return np.array(image), np.array(mask)
    
    def __len__(self):
        return len(self.data_images)
    
def make_dataset(image_dir: str,
                 mask_dir: str) -> Type[torch.utils.data.Dataset]:
    """
    Make pytorch Dataset for given task.
    Read the image using the PIL library and return it as an np.array.

    Args:
        image_dir (str): path of image folder. The image file and mask file must have the same name.
        mask_dir (str): path of mask folder. The image file and mask file must have the same name.

    Returns:
        torch.Dataset: pytorch Dataset
    """
        
    dataset = CustomDataset(image_dir=image_dir,
                            mask_dir=mask_dir,
                            transform=None)
        
    return dataset