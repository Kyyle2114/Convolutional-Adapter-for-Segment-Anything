#!/bin/bash

TRAIN_IMAGE_DIR=dataset/train/image
TRAIN_MASK_DIR=dataset/train/mask
VAL_IMAGE_DIR=dataset/val/image
VAL_MASK_DIR=dataset/val/mask

python3 run_cad.py \
    --batch_size 4 \
    --port 1234 \
    --dist False \
    --seed 21 \
    --model_type vit_b \
    --checkpoint sam_vit_b.pth \
    --epoch 20 \
    --lr 2e-4 \
    --project_name Conv_Adapter_Experiments \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --train_mask_dir $TRAIN_MASK_DIR \
    --val_image_dir $VAL_IMAGE_DIR \
    --val_mask_dir $VAL_MASK_DIR \

python3 run_sa.py \
    --batch_size 4 \
    --port 1234 \
    --dist False \
    --seed 21 \
    --model_type vit_b \
    --checkpoint sam_vit_b.pth \
    --epoch 20 \
    --lr 2e-4 \
    --project_name Conv_Adapter_Experiments \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --train_mask_dir $TRAIN_MASK_DIR \
    --val_image_dir $VAL_IMAGE_DIR \
    --val_mask_dir $VAL_MASK_DIR \