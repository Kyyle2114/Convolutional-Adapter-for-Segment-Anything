import warnings
warnings.filterwarnings('ignore')

from segment_anything_cad import sam_model_registry
from segment_anything_cad.utils import *

import torch.nn as nn 

from torch.utils.data import DataLoader

device = 'cuda:0'

sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
sam.to(device)

sam = save_weight.load_partial_weight(
    model=sam,
    load_path='checkpoints/May11_190102.pth',
    dist=True
)

val_set = dataset.make_dataset(
    image_dir='datasets/test/image',
    mask_dir='datasets/test/mask'
)

val_loader = DataLoader(
    val_set, 
    batch_size=2, 
    shuffle=False
)

bceloss = nn.BCELoss()
iouloss = iou_loss_torch.IoULoss()

val_bce_loss, val_iou_loss, val_dice, val_iou = trainer.model_evaluate(
    model=sam,
    data_loader=val_loader,
    criterion=[bceloss, iouloss],
    device=device,
)

print(f'val_bce_loss: {val_bce_loss:.5f}, val_iou_loss: {val_iou_loss:.5f}, val_dice: {val_dice:.5f}, val_iou: {val_iou:.5f} \n')
