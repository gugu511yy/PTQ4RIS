# PTQ4RIS: Post-Training Quantization for Referring Image Segmentation

## Abstract

[//]: # (This repository contains the code for the paper **"PTQ4RIS: An Effective and Efficient Post-Training Quantization Framework for Referring Image Segmentation"**.)

Referring Image Segmentation (RIS), aims to segment the object referred by a given sentence in an image by understanding both visual and linguistic information. However, existing RIS methods tend to explore top-performance models, disregarding considerations for practical applications on resources-limited edge devices. This oversight poses a significant challenge for on-device RIS inference. To this end, we propose an effective and efficient post-training quantization framework termed PTQ4RIS. Specifically, we first conduct an in-depth analysis of the root causes of performance degradation in RIS model quantization and propose dual-region quantization (DRQ) and reorder-based outlier-retained quantization (RORQ) to address the quantization difficulties in visual and text encoders. Extensive experiments on three benchmarks with different bits settings (from 8 to 4 bits) demonstrates its superior performance. Importantly, we are the first PTQ method specifically designed for the RIS task, highlighting the feasibility of PTQ in RIS applications. 

Paper: [arxiv](https://arxiv.org/abs/2409.17020) / Video: [youtube](https://www.youtube.com/watch?v=EGy-PD7rRfk)

[//]: # (![PTQ4RIS Framework]&#40;image.png&#41;)

## Quantization Results of PTQ4RIS
### MIoU Results Across Different Datasets.
| **Bit-width** | **Method**                | **RefCOCO val** | **RefCOCO testA** | **RefCOCO testB** | **RefCOCO+ val** | **RefCOCO+ testA** | **RefCOCO+ testB** | **G-Ref val(U)** | **G-Ref test(U)** | **G-Ref val(G)** |
|:--------------:|:-------------------------:|:----------------:|:------------------:|:------------------:|:------------------:|:--------------------:|:------------------:|:-----------------:|:------------------:|:-----------------:|
| FP        | baseline (LAVT)            | 74.31            | 76.63              | 70.61             | 66.69             | 71.47               | 60.01               | 65.91             | 66.01              | 64.08             |
| W8A8         | **PTQ4RIS (Ours)**       | **73.54**        | **76.24**          | **70.21**         | **66.42**         | **71.32**           | **59.76**           | **65.47**         | **65.62**          | **63.93**         |
| W6A6          | **PTQ4RIS (Ours)**       | **72.85**        | **75.37**          | **68.62**         | **65.10**         | **69.70**           | **58.55**           | **65.02**         | **65.31**          | **63.66**         |
| W4A8          | **PTQ4RIS (Ours)**       | **72.62**        | **75.12**          | **68.47**         | **64.29**         | **69.50**           | **57.45**           | **63.87**         | **63.96**          | **62.08**         |
| W4A4          | **PTQ4RIS (Ours)**       | **69.53**        | **71.67**          | **65.35**         | **61.25**         | **65.77**           | **54.29**           | **60.60**         | **60.86**          | **59.92**         |

### OIoU Results Across Different Datasets.
| **Bit-width** | **Method**                | **RefCOCO val** | **RefCOCO testA** | **RefCOCO testB** | **RefCOCO+ val** | **RefCOCO+ testA** | **RefCOCO+ testB** | **G-Ref val(U)** | **G-Ref test(U)** | **G-Ref val(G)** |
|:--------------:|:-------------------------:|:----------------:|:------------------:|:------------------:|:------------------:|:--------------------:|:------------------:|:-----------------:|:------------------:|:-----------------:|
| FP        | baseline (LAVT)            |      72.72      |        75.74       |        68.56       |        63.38       |        68.73        |        56.08        |        62.65      |        64.10       |        60.85      |
| W8A8          | **PTQ4RIS (Ours)**       |     **72.33**    |      **75.38**     |      **68.26**     |      **63.32**     |      **68.72**      |      **56.07**      |      **62.46**    |      **63.84**     |      **60.80**    |
| W6A6          | **PTQ4RIS (Ours)**       |     **71.90**    |      **75.03**     |      **67.40**     |      **62.50**     |      **67.52**      |      **55.42**      |      **62.01**    |      **63.47**     |      **60.61**    |
| W4A8          | **PTQ4RIS (Ours)**       |     **71.48**    |      **74.60**     |      **66.53**     |      **61.43**     |      **67.17**      |      **54.39**      |      **61.29**    |      **62.38**     |      **59.68**    |
| W4A4          | **PTQ4RIS (Ours)**       |     **69.21**    |      **72.00**     |      **64.11**     |      **59.44**     |      **64.26**      |      **51.87**      |      **58.98**    |      **60.37**     |      **58.27**    |


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
Code in this repository is built upon several public repositories, thanks for those great works! Specifically,
* Quantization model is built upon [LAVT](https://github.com/yz93/LAVT-RIS) 
* The quantization code is partly built upon [PTQ4ViT](https://github.com/hahnyuan/PTQ4ViT) and [PD-Quant](https://github.com/hustvl/PD-Quant).

### License
This repository is released under MIT License (see LICENSE file for details).