# Task Description
The language modeling task requires the language model to predict the next text based on the previous information. 
In this task, we consider a long-context language modeling where the context length is prolonged to 8,192. 
Perplexity (PPL) is used to serve as the evaluation metric.

<div align="center">

Model           | PPL↓
--------------- | :--------: 
LongShort       | **15.52**
S4D             | 15.78
FlashAttention  | 16.02
local           | 19.73 
ABC             | 29.13

</div>


# Dataset Statistics
We use **PG-19** dataset for evaluating language models. 
PG-19 consists of books extracted from the Project Gutenberg books library. 
The train/valid/test sets contain 28,602/50/100 books, respectively. 

# Baseline and Reproducibility
We use [GPT-2](https://github.com/facebookresearch/fairseq/tree/main/examples/language_model) as our backbone model. 
To easily reproduce the results, you can follow the next steps.

## Building Environment
```python
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

## Data Preparation
Integrate the downloaded [PG-19](https://github.com/deepmind/pg19) dataset into `train.txt`, `vaild.txt`, and `text.txt`. We use GPT-2 tokenizer to convert text into tokens as follows:
```python
python preprocess.py data/PG-19/train.txt
python preprocess.py data/PG-19/valid.txt
python preprocess.py data/PG-19/test.txt
```
Then we use fairseq to binarize the data:
```bash
TEXT=data/PG-19/
fairseq-preprocess \
     --only-source \
     --trainpref $TEXT/train.txt \
     --validpref $TEXT/valid.txt \
     --testpref $TEXT/test.txt \
     --destdir data-bin/PG-19 \
     --workers 20
```

## Training
Refer to the [fairseq language model](https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/README.md), we use 8×80GB A100 GPU to train a language model as follows:
```bash
fairseq-train --task language_modeling \
  data-bin/PG-19 \
  --save-dir checkpoints \
  --arch transformer_lm_gpt --share-decoder-input-output-embed \
  --dropout 0 \
  --optimizer adam --adam-betas '(0.9, 0.999)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 8192 --sample-break-mode none \
  --max-tokens 8192 --update-freq 2 \
  --max-update 125000 \
  --seed 19260817
```

## Test
```bash
fairseq-eval-lm data-bin/PG-19 \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 16 \
    --tokens-per-sample 8192 \
    --context-window 0
```
