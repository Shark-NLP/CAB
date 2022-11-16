# Task Description
In this task, we aims to convert low-resolution (16 × 16) face images into high-resolution (128 × 128) images.
Peak Signal-to-Noise Ratio (PSNR) and Structural SIMilarity (SSIM) are selected to evaluate models from different aspects.

<div align="center">

Model        |   PSNR↑  | SSMI↑
------       |  ------  | ---
cosFormer    | **23.53**| **0.690**
LARA         | 23.35    | 0.685 
S4D          | 23.35    | 0.682 
Performer    | 23.34    | 0.682 
local        | 23.33    | 0.682 
LongShort    | 23.28    | 0.681 
Nyströmformer| 23.20    | 0.679 
ProbSparse   | 22.98 	| 0.667 
ABC	         | 22.54    | 0.635 

</div>

# Dataset Statistics
We train face super-resolution models on Flickr-Faces-HQ (FFHQ) and conduct evaluation on CelebA-HQ.
FFHQ consists of 70,000 high-quality PNG images at 1024×1024 resolution. 
CelebA-HQ contains 30,000 high-quality celebrity faces.

# Baseline and Reproducibility
We use an unofficial implementation [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/ef9b943b573328d7a5ddb1a0c2abd168b91610dc) introduced in "Image Super-Resolution via Iterative Refinement" as our baseline model. 
To easily reproduce the results, you can follow the next steps.

## Building Environment
```python
pip install -r requirement.txt
```

## Data Preparation
Download the [FFHQ](https://github.com/NVlabs/ffhq-dataset) and [CelebA-HQ](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256), and prepare it in **LMDB** or **PNG** format using script.
```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```

change the datasets config to your data path and image resolution: 
```json
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "lmdb", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
    }
},
```

## Training
We use 1×80GB A100 GPU to train models. You can train the model with the following scripts:
```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c sr_sr3_16_128.json
```

## Test
Due to the large variance of training results, in order to obtain fair results, we adopt the checkpoint average method to obtain `Average_gen.pth` from the last five checkpoints. 
You can get `Average_gen.pth` with the following command:
```python
python checkpoint_average.py checkpoint/
```

Change the config to resume checkpoint:
```json
"path": { //set the path
        "log": "logs",
        "tb_logger": "tb_logger",
        "results": "results",
        "checkpoint": "checkpoint",
        "resume_state": "checkpoint/Average" //pretrain model or training state
    },
```

Finally, use the following command to test：
```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c sr_sr3_16_128.json

```
