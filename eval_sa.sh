#!/bin/bash

echo SA_EVALUATION >> SA_EVAL.txt

for tag in 1 2 3 4 5 6 7 8 9 10 
do 
    echo TEST DATASET ${tag} >> SA_EVAL.txt
    python3 eval_sa.py --batch_size 4 --seed 21 --model_type vit_b --checkpoint medsam_vit_b.pth --dataset ${tag} >> SA_EVAL.txt
done 
