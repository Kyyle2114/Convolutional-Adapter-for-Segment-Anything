#!/bin/bash

BATCH_SIZE=4
SEED=21
MODEL_TYPE=vit_b
TEST_IMAGE_DIR=dataset/test/image
TEST_MASK_DIR=dataset/test/mask

echo CONV_ADAPTER_EVALUATION >> CAD_EVAL.txt

python3 eval_cad.py \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --model_type $MODEL_TYPE \
    --checkpoint checkpoints/sam_cad.pth \
    --test_image_dir $TEST_IMAGE_DIR \
    --test_mask_dir $TEST_MASK_DIR \
    >> CAD_EVAL.txt

echo SAM_ADAPTER_EVALUATION >> SA_EVAL.txt

python3 eval_sa.py \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --model_type $MODEL_TYPE \
    --checkpoint checkpoints/sam_sa.pth \
    --test_image_dir $TEST_IMAGE_DIR \
    --test_mask_dir $TEST_MASK_DIR \
    >> SA_EVAL.txt