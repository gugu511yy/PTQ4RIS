# PTQ4RIS: Post-Training Quantization for Referring Image Segmentation

## Abstract

[//]: # (This repository contains the code for the paper **"PTQ4RIS: An Effective and Efficient Post-Training Quantization Framework for Referring Image Segmentation"**.)

Referring Image Segmentation (RIS), aims to segment the object referred by a given sentence in an image by understanding both visual and linguistic information. However, existing RIS methods tend to explore top-performance models, disregarding considerations for practical applications on resources-limited edge devices. This oversight poses a significant challenge for on-device RIS inference. To this end, we propose an effective and efficient post-training quantization framework termed PTQ4RIS. Specifically, we first conduct an in-depth analysis of the root causes of performance degradation in RIS model quantization and propose dual-region quantization (DRQ) and reorder-based outlier-retained quantization (RORQ) to address the quantization difficulties in visual and text encoders. Extensive experiments on three benchmarks with different bits settings (from 8 to 4 bits) demonstrates its superior performance. Importantly, we are the first PTQ method specifically designed for the RIS task, highlighting the feasibility of PTQ in RIS applications. 

Paper : [arxiv](https://github.com/gugu511yy/PTQ4RIS) / Video : [video](https://github.com/gugu511yy/PTQ4RIS)

[//]: # (![PTQ4RIS Framework]&#40;image.png&#41;)

## Setting Up
### Preliminaries
The code has been verified to work with PyTorch v1.7.1 and Python 3.7.
1. Clone this repository.
2. Change directory to root of this repository.
```shell
git clone git@github.com:gugu511yy/PTQ4RIS.git
```
### Package Dependencies
1. Create a new Conda environment with Python 3.7 then activate it:
```shell
conda create -n ptq4ris python==3.7
conda activate ptq4ris
```

2. Install PyTorch v1.7.1 with a CUDA version that works on your cluster/machine (CUDA 10.2 is used in this example):
```shell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
```

3. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```

### Datasets
1. Follow instructions in the `./refer` directory to set up subdirectories
and download annotations.
This directory is a git clone (minus two data files that we do not need)
from the [refer](https://github.com/lichengunc/refer) public API.

2. Download images from [COCO](https://cocodataset.org/#download).
Please use the first downloading link *2014 Train images [83K/13GB]*, and extract
the downloaded `train_2014.zip` file to `./refer/data/images/mscoco/images`.

### The Initialization Weights for Training
1. Create the `./pretrained_weights` directory where we will be storing the weights.
```shell
mkdir ./pretrained_weights
```
2. Download [pre-trained classification weights of
the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth),
and put the `pth` file in `./pretrained_weights`.
These weights are needed for training to initialize the model.

### Trained Weights of LAVT for Testing
1. Create the `./checkpoints` directory where we will be storing the weights.
```shell
mkdir ./checkpoints
```
2. Download the FP model weights using links below and put them in `./checkpoints`.

| RefCOCO | RefCOCO+ | G-Ref (UMD) | G-Ref (Google) |
|:-----:|:-----:|:-----:|:-----:|
|[log](https://drive.google.com/file/d/1YIojIHqe3bxxsWOltifa2U9jH67hPHLM/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1xFMEXr6AGU97Ypj1yr8oo00uObbeIQvJ/view?usp=sharing)|[log](https://drive.google.com/file/d/1Z34T4gEnWlvcSUQya7txOuM0zdLK7MRT/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1HS8ZnGaiPJr-OmoUn4-4LVnVtD_zHY6w/view?usp=sharing)|[log](https://drive.google.com/file/d/14VAgahngOV8NA6noLZCqDoqaUrlW14v8/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/14g8NzgZn6HzC6tP_bsQuWmh5LnOcovsE/view?usp=sharing)|[log](https://drive.google.com/file/d/1JBXfmlwemWSvs92Rky0TlHcVuuLpt4Da/view?usp=sharing) &#124; [weights](https://drive.google.com/file/d/1IJeahFVLgKxu_BVmWacZs3oUzgTCeWcz/view?usp=sharing)|

### Testing   
```shell
python ptq4ris.py  --model lavt_one --swin_type base --dataset refcoco  --split val --resume ./checkpoints/lavt_one_8_cards_ImgNet22KPre_swin-base-window12_refcoco_adamw_b32lr0.00005wd1e-2_E40.pth  --workers 0 --ddp_trained_weights --window12 --img_size 480 --num_samples 32 --n_bits_w 8 --n_bits_a 8   

```
### Acknowledgments
Code in this repository is built upon several public repositories. Specifically,
* This repo is built upon [LAVT](https://github.com/yz93/LAVT-RIS) 
* The quantization code is partly built upon [PTQViT](https://github.com/hahnyuan/PTQ4ViT) and [PD-Quant](https://github.com/hustvl/PD-Quant).

### License
This repository is released under MIT License (see LICENSE file for details).