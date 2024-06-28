#!/bin/bash

python3 run_cad.py --batch_size 4 --port 1234 --dist False --seed 21 --model_type vit_b --checkpoint sam_vit_b.pth --epoch 20 --lr 2e-4 --project_name CAD-SA-Comparison

python3 run_sa.py --batch_size 4 --port 1234 --dist False --seed 21 --model_type vit_b --checkpoint sam_vit_b.pth --epoch 20 --lr 2e-4 --project_name CAD-SA-Comparison