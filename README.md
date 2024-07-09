# CAD : Memory Efficient Convolutional Adapter for Segment Anything

JooHyeok Kim, SeoHwan Yun, JoonHyeon Song, SeongHo Yun

AIaaS Laboratory, Kwangwoon University

## Abstract 

> Paper Abstract

## Overview

![alt text](assets/image.png)

We proposed memory efficient convolutional adapter for Segment Anything(SAM). 

## Enviornment

```bash 
pip install -r requirements.txt 
```

## How to Use 

- Move to ```code/``` directory
- [Download the checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) for Segment Anything corresponding to the model type(e.g., ```vit_b```).
- Inside ```run.sh```, set the appropriate dataset path (train, validation)
- You can check the meaning of each argument in argparse at the top of ```run_cad.py```, ```run_sa.py```
- Running ```./run.sh``` will train the both model

All training processes are logged using **WandB(Weights & Biases)**. To modify the wandb project name, modify the ```--project_name``` argument.

After model training is complete, the model weights are saved under the ```checkpoints/``` directory. The weight file name is saved based on the execution time (e.g., ```May21_172910.pth```). Rename this file to ```sam_cad.pth``` and ```sam_sa.pth```.

If you want to evaluate the trained model, set the test dataset path inside ```eval.sh```, then run the ```./eval.sh```.