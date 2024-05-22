#!/bin/bash

python3 run.py --batch_size 4 --port 1234 --dist True --seed 21 --model_type vit_h --checkpoint sam_vit_h.pth --epoch 5 --lr 2e-4 --project_name Conv-Adapter-SAM