import warnings
warnings.filterwarnings('ignore')

from segment_anything import sam_model_registry
from segment_anything.utils import *

import torch
import torch.nn as nn 

import torch.distributed as dist
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import os
import argparse
from datetime import datetime
from torchinfo import summary

CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--port', type=int, default=2024)
    parser.add_argument('--dist', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--model_type', type=str, default='vit_t')
    parser.add_argument('--checkpoint', type=str, default='sam_vit_t.pt')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--local_rank', type=int)
    
    return parser

### Fine-tuning SAM ###
def main(rank, opts) -> str:
    """
    Model fine-tuning
    
    Returns:
        str: Save path of model checkpoint 
    """
    seed.seed_everything(opts.seed)  
    
    set_dist.init_distributed_training(rank, opts)
    local_gpu_id = opts.gpu
    
    ### checkpoint set ### 
    run_time = datetime.now()
    run_time = run_time.strftime("%b%d_%H%M%S")
    file_name = run_time + '.pth'
    save_path = os.path.join(CHECKPOINT_DIR, file_name)

    ### dataset & dataloader ### 
    train_set = dataset.make_dataset(
        image_dir='datasets/train/image',
        mask_dir='datasets/train/mask'
    )
    
    val_set = dataset.make_dataset(
        image_dir='datasets/test/image',
        mask_dir='datasets/test/mask'
    )
    
    if opts.dist:
        train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
        batch_sampler_train = BatchSampler(train_sampler, opts.batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=opts.num_workers)
    
    if not opts.dist:
        train_loader = DataLoader(
            train_set, 
            batch_size=opts.batch_size, 
            shuffle=True, 
            num_workers=opts.num_workers
        )

    val_loader = DataLoader(
        val_set, 
        batch_size=opts.batch_size, 
        shuffle=False
    )
    
    ### SAM config ### 
    sam_checkpoint = opts.checkpoint
    model_type = opts.model_type

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.cuda(local_gpu_id)
    
    ### Trainable config ###
    for _, p in sam.image_encoder.named_parameters():
        p.requires_grad = False
        
    for _, p in sam.prompt_encoder.named_parameters():
        p.requires_grad = False

    # fine-tuning mask decoder         
    for _, p in sam.mask_decoder.named_parameters():
        p.requires_grad = True
    
    # print model info 
    print()
    print('=== MODEL INFO ===')
    summary(sam)
    print()

    sam = DistributedDataParallel(module=sam, device_ids=[local_gpu_id])    

    ### train config ###  
    bceloss = nn.BCELoss().to(local_gpu_id)
    iouloss = iou_loss_torch.IoULoss().to(local_gpu_id)

    EPOCHS = opts.epoch
    lr = opts.lr
    # EarlyStopping : Determined based on the validation loss. Lower is better(mode='min').
    es = trainer.EarlyStopping(patience=EPOCHS//2, delta=0, mode='min', verbose=True)
    optimizer = torch.optim.AdamW(sam.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=len(train_loader), 
        eta_min=0,
        last_epoch=-1
    )

    max_loss = np.inf 
    
    for epoch in range(EPOCHS):
        
        if opts.dist:
            train_sampler.set_epoch(epoch)
        
        ### model train / validation ###
        train_bce_loss, train_iou_loss, train_dice, train_iou = trainer.model_train(
            model=sam.module,
            data_loader=train_loader,
            criterion=[bceloss, iouloss],
            optimizer=optimizer,
            device=local_gpu_id,
            scheduler=scheduler
        )
        
        if opts.dist:
            dist.all_reduce(train_bce_loss, op=dist.ReduceOp.SUM)            
            train_bce_loss = train_bce_loss.item() / dist.get_world_size()
            
            dist.all_reduce(train_iou_loss, op=dist.ReduceOp.SUM)            
            train_iou_loss = train_iou_loss.item() / dist.get_world_size()
            
            dist.all_reduce(train_dice, op=dist.ReduceOp.SUM)
            train_dice = train_dice.item() / dist.get_world_size()
            
            dist.all_reduce(train_iou, op=dist.ReduceOp.SUM)
            train_iou = train_iou.item() / dist.get_world_size()
            
        if not opts.dist:
            train_bce_loss = train_bce_loss.item()
            train_iou_loss = train_iou_loss.item()
            train_dice = train_dice.item()
            train_iou = train_iou.item()
        
        if opts.rank == 0:
            val_bce_loss, val_iou_loss, val_dice, val_iou = trainer.model_evaluate(
                model=sam.module,
                data_loader=val_loader,
                criterion=[bceloss, iouloss],
                device=f"cuda:{local_gpu_id}"
            )
            
            val_loss = val_bce_loss + val_iou_loss
            
            # Check EarlyStopping
            es(val_loss)
            if es.early_stop:
                print(f'Model checkpoint saved at: {save_path} \n')
                break
        
            ### Save best model ###
            if val_loss < max_loss:
                print(f'[INFO] val_loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save model.')
                max_loss = val_loss
                _ = save_weight.save_partial_weight(model=sam, save_path=save_path)
            
            ### print current loss / metric ###
            print(f'epoch {epoch+1:02d}, bce_loss: {train_bce_loss:.5f}, iou_loss: {train_iou_loss:.5f}, dice: {train_dice:.5f}, iou: {train_iou:.5f},', end=' ')
            print(f'val_bce_loss: {val_bce_loss:.5f}, val_iou_loss: {val_iou_loss:.5f}, val_dice: {val_dice:.5f}, val_iou: {val_iou:.5f} \n')
    
    print(f'Model checkpoint saved at: {save_path} \n') 

if __name__ == '__main__': 

    parser = argparse.ArgumentParser('Fine-tuning SAM', parents=[get_args_parser()])
    opts = parser.parse_args() 
    
    if opts.dist:
        opts.ngpus_per_node = torch.cuda.device_count()
        opts.gpu_ids = list(range(opts.ngpus_per_node))
        
    if not opts.dist:
        opts.ngpus_per_node = 1
        opts.gpu_ids = [0]
    
    opts.num_workers = opts.ngpus_per_node * 4

    torch.multiprocessing.spawn(
        main,
        args=(opts,),
        nprocs=opts.ngpus_per_node,
        join=True
    )
    
    print('=== DONE === \n')    