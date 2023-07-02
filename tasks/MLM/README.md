# Task Description
masked language modeling uses full context to predict masked tokens.
We use Perplexity (PPL) as evaluation metric and set the context length to 2,048.

<div align="center">

Model        | PPL↓
---------    | ---:
FlashAttention | **3.26**
LongShort    | 3.38
local        | 4.18
cosFormer    | 5.25
Nyströmformer| 6.32
LARA	       | 6.45
ABC	         | 10.50
S4D          | 25.34
Performer    | 111.73
ProbSparse	 | 186.35

</div>

# Dataset Statistics
We use **PG-19** dataset for masked language modeling task. PG-19 consists of books extracted from the Project Gutenberg books library. The train/valid/test sets contain 28,602/50/100 books, respectively. 

# Baseline and Reproducibility
We use [RoBERTa](https://github.com/facebookresearch/fairseq/tree/main/examples/language_model) as our backbone model. To easily reproduce the results, you can follow the next steps.

## Building Environment
```python
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

## Data Preparation
Refer to LM task. 

## Training and test
Refer to the [fairseq roberta](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md), we use 8×80GB A100 GPU to train a model as follows: 
```bash
DATA_DIR=data-bin/PG19
fairseq-hydra-train -m --config-dir configs --config-name mlm task.data=$DATA_DIR
```
We take model's PPL on vaild set as results.
