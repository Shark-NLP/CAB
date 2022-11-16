# Task Description
This task is one that makes a comprehensive summary of multiple input documents without losing important information. 
We consider multi-document summarization task for long sequence modeling. 
To make the task more challenging, we set the maximum source text and target summary lengths to be 4,096 and 400, respectively.
We take ROUGE(R-N) as task metric.

<div align="center">
  
<b>Noncausal Self</b>

Model           | R-1↑    | R-2↑     | R-L↑
--------------  | ------- | -------- | ---
local           | **38.50**   | **10.54**    | **35.39** 
Performer       | 34.85   | 6.54     | 31.88 
cosFormer       | 34.77   | 6.34     | 31.74 
ProbSparse      | 34.62   | 6.36     | 31.64 
Nyströmformer	  | 34.45   | 6.30     | 31.56 
LongShort       | 34.35   | 6.41     | 31.55 
LARA            | 34.03   | 6.23     | 31.23 
ABC             | 33.80   | 6.07     | 30.98 
S4D             | -	      | -        | -

</div>

<div align="center">

<b>Causal Self</b>
  
Model           | R-1↑    | R-2↑     | R-L↑
--------------  | ------- | -------- | ---
S4D             | **34.90**   | **6.65**     | **31.98**
LongShort       | 33.55   | 6.27     | 30.71 
local           | 33.50   | 6.27     | 30.74
ABC             | 30.17   | 5.48     | 27.92 

</div>

<div align="center">
  
<b>Causal Cross</b>
  
Model           | R-1↑    | R-2↑     | R-L↑
--------------  | ------- | -------- | ---
ABC             | **32.22**   | **5.55**     | **29.53** 
Performer       | 27.22   | 3.88     | 25.21 

</div>

# Dataset Statistics
We use Multi-News dataset for evaluation. 
The source and target texts of Multi-News contain ~2300 and ~280 tokens on average, respectively.

# Baseline and Reproducibility
We use Transformer as backbone model. To easily reproduce the results, you can follow the next steps.

## Building Environment
We conduct experiments with [ParaGen](https://github.com/bytedance/ParaGen/tree/main/examples/summarization). 
The setup of ParaGen refers to [document](https://github.com/bytedance/ParaGen).

## Data Preparation
Our preprossed Multi-News dataset is downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1DQdhPaxfZnclRwhjlRieFpnQkhe1s-DZ?usp=sharing). The origin dataset is on [Huggingface](https://huggingface.co/datasets/multi_news).
The mutli-news dataset is preprocessed with 
```bash
paragen-preprocess --config configs/preprocess.yaml
```
which tokenizes sentences with `bart-base` tokenizer and transform the tokens into index.

## Training
We use 4×80GB A100 GPU to train a Transformer model as follows:
```shell
cd examples/summarization
python -m torch.distributed.launch --nproc_per_node 4 paragen-run --config train.yaml --lib efficient_transformers --env.fp16 True
```
For faster training with `fp16`, please specify `--env.fp16 True`.

## Test
```shell
paragen-run --config eval.yaml --lib summ,efficient_transformers --env.fp16 True
```
