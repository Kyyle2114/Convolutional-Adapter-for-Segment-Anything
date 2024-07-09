#!/bin/bash

TEST_IMAGE_DIR=dataset/test/image
TEST_MASK_DIR=dataset/test/mask

echo CONV_ADAPTER_EVALUATION >> CAD_EVAL.txt

python3 eval_cad.py \
    --batch_size 4 \
    --seed 21 \
    --model_type vit_b \
    --checkpoint sam_vit_b.pth \
    --adapter_checkpoint checkpoints/sam_cad.pth \
    --test_image_dir $TEST_IMAGE_DIR \
    --test_mask_dir $TEST_MASK_DIR \
    >> CAD_EVAL.txt

echo SAM_ADAPTER_EVALUATION >> SA_EVAL.txt

python3 eval_sa.py \
    --batch_size 4 \
    --seed 21 \
    --model_type vit_b \
    --checkpoint sam_vit_b.pth \
    --adapter_checkpoint checkpoints/sam_sa.pth \
    --test_image_dir $TEST_IMAGE_DIR \
    --test_mask_dir $TEST_MASK_DIR \
    >> SA_EVAL.txt