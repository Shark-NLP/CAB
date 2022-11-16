# Task Description
Long sequence time-series forecasting is to predict long-term future behavior based on past states. 
This task evaluates models on three datasets, including Electricity Transformer Temperature (ETT), Electricity Consuming Load (ECL), and Weather.
Following [(Zhou et al., 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325), we also conduct `univariate` and `multivariate` evaluations and average their Mean Square Error (MSE) and Mean Absolute Error (MAE) to obtain the final scores.

<div align="center">

 Model     | ETT-MSE↓ | ETT-MAE↓ | ECL-MSE↓ | ECL-MAE↓ | Weather-MSE↓ | Weather-MAE↓ 
 -------   | ------- | ------- | ------- | ------- | -------     | -------     
 ABC       | **1.147**   | **0.809**   | 0.489   | 0.520   | **0.519**       | **0.527**
 Performer | 1.254   | 0.819   | **0.463**   | **0.508**   | 0.881       | 0.722
 cosFormer | 1.219   | 0.823   | 0.474   | 0.509   | 0.632       | 0.590

</div>

# Dataset Statistics
## ETT
ETT contains three sub-datasets: ETTh1, ETTh2, ETTm1. Both ETTh1 and ETTh2 contain 17,420 hour-level data points (Approximately 2 year * 365 days * 24 hour). ETTm1 contains 69,681 quarter-level data points (Approximately 2 year * 365 days * 24 hour * 4). All the data is stored in the `.csv` file format. The first line (8 columns) is the horizontal header and includes "date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", and "OT". The detailed meaning of each column name is shown in Table 1.

| Field | date | HUFL | HULL | MUFL | MULL | LUFL | LULL | OT |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Description | The recorded **date** |**H**igh **U**se**F**ul **L**oad | **H**igh **U**se**L**ess **L**oad | **M**iddle **U**se**F**ul **L**oad | **M**iddle **U**se**L**ess **L**oad | **L**ow **U**se**F**ul **L**oad | **L**ow **U**se**L**ess **L**oad | **O**il **T**emperature (target) |

<p align="center"><b>Table 1.</b> Description for each columm.</p>

We first calculate the scores of the model on ETTh1, ETTh2, ETTm1 respectively. `Then we average the scores of three sub-datasets to obtain the score on ETT dataset.`

$$MAE_{ETT} = \frac{(MAE_{ETTh1-u} + MAE_{ETTh2-u} + MAE_{ETTm1-u} + MAE_{ETTh1-m} + MAE_{ETTh2-m} + MAE_{ETTm1-m})}{6}$$
where $u$ denotes univariate and $m$ denotes multivariate.

## ECL
ECL collects the 2-year electricity consumption (Kwh) of 321 clients and contains 26,304 hour-level data points. We set "MT_320" as the target value.

## Weather
Weather contains local climatological data for nearly 1,600 U.S. locations, 4 years from 2010 to 2013, where data points are collected every 1 hour. Each data point consists of the target value "wet bulb" and 11 climate features. 

# Baseline and Reproducibility
We use Informer as our backbone model. To easily reproduce the results, you can follow the next steps.

## Building Environment
```python
pip install -r requirement.txt
```

## Download Data
The ETT, ECL and Weather datasets can be downloaded in the repo [Informer2020](https://github.com/zhouhaoyi/Informer2020). The required data files should be put into `data/ETT/` folder.

## Training and Test
We use 1×80GB A100 GPU to train models. You can train and test the model with the following commands:
```shell
bash scripts/ETTh1.sh
bash scripts/ETTh2.sh
bash scripts/ETTm1.sh
bash scripts/ECL.sh
bash scripts/WTH.sh
```

## Calculating Results
We provide a python file to quickly obtain the experimental results. 
```python
python get_results.py results/ 
```
Our detail results are shown in `results.xlsx`.
