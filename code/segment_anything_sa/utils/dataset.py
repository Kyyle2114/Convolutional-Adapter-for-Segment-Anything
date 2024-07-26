import os
import numpy as np 
import cv2
from typing import Type

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.data_images = sorted(os.listdir(image_dir))
        self.mask_images = sorted(os.listdir(mask_dir))
        
        assert len(self.data_images) == len(self.mask_images), "<Error> The number of images and the number of masks must be the same."

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data_images[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_images[idx])  

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        mask[mask != 0] = 1 

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            
        return image, mask
    
    def __len__(self):
        return len(self.mask_images)
    
def make_dataset(
    image_dir: str,
    mask_dir: str,
    transform = None
) -> Type[torch.utils.data.Dataset]:
    """
    Make pytorch Dataset for given task.
    Read the image using the opencv library and return it as an np.array.

    Args:
        image_dir (str): path of image folder. The image file and mask file must have the same name.
        mask_dir (str): path of mask folder. The image file and mask file must have the same name.
        transform (albumentations transform): The albumentations transforms to be applied to images and masks. Defaults to None.

    Returns:
        torch.Dataset: pytorch Dataset
    """
        
    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform
    )
        
    return dataset