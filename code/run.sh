#!/bin/bash

BATCH_SIZE=4
PORT=1234
DIST=False
SEED=21
MODEL_TYPE=vit_b
CHECKPOINT=sam_vit_b.pth
EPOCH=20
LR=2e-4
PROJECT_NAME=Conv_Adapter_Experiments
TRAIN_IMAGE_DIR=dataset/train/image
TRAIN_MASK_DIR=dataset/train/mask
VAL_IMAGE_DIR=dataset/val/image
VAL_MASK_DIR=dataset/val/mask

python3 run_cad.py \
    --batch_size $BATCH_SIZE \
    --port $PORT \
    --dist $DIST \
    --seed $SEED \
    --model_type $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --epoch $EPOCH \
    --lr $LR \
    --project_name $PROJECT_NAME \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --train_mask_dir $TRAIN_MASK_DIR \
    --val_image_dir $VAL_IMAGE_DIR \
    --val_mask_dir $VAL_MASK_DIR \

python3 run_sa.py \
    --batch_size $BATCH_SIZE \
    --port $PORT \
    --dist $DIST \
    --seed $SEED \
    --model_type $MODEL_TYPE \
    --checkpoint $CHECKPOINT \
    --epoch $EPOCH \
    --lr $LR \
    --project_name $PROJECT_NAME \
    --train_image_dir $TRAIN_IMAGE_DIR \
    --train_mask_dir $TRAIN_MASK_DIR \
    --val_image_dir $VAL_IMAGE_DIR \
    --val_mask_dir $VAL_MASK_DIR \