import warnings
warnings.filterwarnings('ignore')

from segment_anything_cad import sam_model_registry
from segment_anything_cad.utils import *

import torch.nn as nn 
from torch.utils.data import DataLoader

import argparse
import albumentations as A

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size allocated to GPU')
    parser.add_argument('--seed', type=int, default=21, help='random seed')
    parser.add_argument('--model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--checkpoint', type=str, default='sam_vit_b.pth', help='SAM model checkpoint')
    parser.add_argument('--test_image_dir', type=str, default='dataset/test/image', help='test dataset image dir')
    parser.add_argument('--test_mask_dir', type=str, default='dataset/test/mask', help='test dataset mask dir')
    
    return parser

def main(opts):
    """
    Model evaluation

    Args:
        opts (argparser): argparser
    """
    seed.seed_everything(opts.seed)
    
    ### dataset & dataloader ### 
    test_set = dataset.make_dataset(
        image_dir=opts.test_image_dir,
        mask_dir=opts.test_mask_dir,
        transform=A.Resize(512, 512)
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=opts.batch_size, 
        shuffle=False
    )
    
    ### SAM config ### 
    device = 'cuda'
    sam_checkpoint = opts.checkpoint
    model_type = opts.model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()
    
    # freezing parameters
    for _, p in sam.named_parameters():
        p.requires_grad = False
        
    ### loss & metric config ###  
    bceloss = nn.BCELoss()
    iouloss = iou_loss_torch.IoULoss()
    
    test_bce_loss, test_iou_loss, test_dice, test_iou = trainer.model_evaluate(
                model=sam,
                data_loader=test_loader,
                criterion=[bceloss, iouloss],
                device=device
            )
    
    print(f'\ntest_bce_loss: {test_bce_loss:.5f} \ntest_iou_loss: {test_iou_loss:.5f} \ntest_dice: {test_dice:.5f} \ntest_iou: {test_iou:.5f} \n')
    
    return
    
if __name__ == '__main__': 
    parser = argparse.ArgumentParser('Evaluation-SAM', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    main(opts)
    
    print('=== DONE === \n')    