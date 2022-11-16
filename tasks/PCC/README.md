# Task Description
This task is to complete a frame of point cloud data when providing partial points.
Chamfer Distance (CD) and F-Score are selected to evaluate each attention. 
For clearer comparison, CD metrics are enlarged by 1,000 times.

<div align="center">

Model       | CD-l1↓ | CD-l2↓ | F-Score↑
-------     | ------ | -----  | ---
ABC         | **7.344**  | **0.227**  | **0.784**
Performer   | 7.368  | 0.235  | 0.783
cosFormer   | 8.673  | 0.298  | 0.704

</div>

# Dataset Statistics
We launch our experiments on these three datasets: **PCN**.
PCN contains pointcloud data with metadata info as well as the raw data with x, y, z coordinate information of each point.
PCN contains pointcloud data from 8 different categories and use `.pcd` as its file format.
The split of training, validation and testing is available inside the [codebase](https://github.com/yuxumin/PoinTr).

# Baseline and Reproducibility
The experiments adopt [PoinTr](https://arxiv.org/abs/2108.08839) as the backbone and its corresponding code is available in this [url](https://github.com/yuxumin/PoinTr).
Just follow the instruction of the [README](https://github.com/yuxumin/PoinTr/blob/master/README.md) file to decompress the datasets and download the necessary library.

## Building Environment
Change the directory to the codebase repository folder and execute the following command.
```python
pip install -r requirement.txt
base ./install.sh
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
Sometimes it may incur error when installing PointNet++ library. Follow the alternative command:
```python
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
python -m pip install -e .
```

## Data Preparation
download [PCN](https://gateway.infinitescript.com/?fileName=ShapeNetCompletion). If not specifying data storage path, put this datasets inside `data/` folder of the codebase.

## Training and Test
The scripts of model training and test are saved in `PoinTr/scripts/`. You can train and test the model with the following commands. For base settings, we use 2×80GB A100 to train models.
```shell
# training
bash ./scripts/dist_train.sh 2 13323 \
--config /path/to/config
--exp_name example
# testing
bash ./scripts/test.sh 0 \
--config path/to/config
--exp_name example \
--ckpts ./pointr_experiments/<path/to/model>/ckpt-best.pth
```
Just replace the config file with the specific model config.
Other supporting arguments like resuming a model or finetune a model can add `--resume` and `--start_ckpts path/to/ckpt` arguments.
