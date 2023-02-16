# Task Description
This task requires models to convert input text sequences, descriptions, or narrations produced by a single speaker, to synthesized audio clips sharing the same timbre as the provided speaker.
We adopt Mean Cepstral Distortion (MCD) and Mel Spectral Distortion (MSD) for objective evaluation.


<table align="center"> 
<tr><th>FastSpeech 2</th><th>Transformer-TTS</th></tr> 
<tr><td> 

Model          | MCD↓        | MSD↓
-------------- | ----------  | ----
S4D            | **3.303**       | **1.905**
ProbSparse     | 3.363       | 1.946 
ABC            | 3.393       | 1.966
cosFormer      | 3.400       | 1.956 
local          | 3.419       | 1.970 
LongShort      | 3.436       | 1.996   
Performer      | 3.437       | 1.983 
LARA           | 3.463       | 2.012
Nyströmformer  | 3.557       | 2.036 

</td><td> 

Model          | MCD↓        | MSD↓
-------------- | ----------  | ----
LongShort      | **3.913**       | **2.136**
local          | 4.015       | 2.164 
S4D            | 4.017       | 2.195
cosFormer      | 4.030       | 2.160 
ProbSparse     | 4.034       | 2.161 
ABC            | 4.085       | 2.204 
Performer      | 4.115       | 2.198 
LARA           | 4.116       | 2.209 
Nyströmformer  | 4.274       | 2.276

</td></tr> 
</table> 

<div align="center">
  
<b>Causal Self</b>

</div>

<table align="center"> 
<tr><th>Transformer-TTS</th></tr> 
<tr><td>

Model          | MCD↓        | MSD↓
-------------- | ----------  | ----
S4D            | **4.030**       | **2.188**
LongShort      | 4.039       | 2.195
ABC            | 4.058       | 2.189
local          | 4.141       | 2.220

</table>

<div align="center">
  
<b>Causal Self</b>

</div>

<table align="center"> 
<tr><th>Transformer-TTS</th></tr> 
<tr><td>

Model          | MCD↓        | MSD↓
-------------- | ----------  | ----
ABC            | **5.780**       | **2.631**
Performer      | 6.635       | 3.053

</table>
 
 
# Dataset Statistics
We incorporates the **LJSpeech** dataset whose audio clips are sampled with 22,050 Hz. 
Under this set of relatively high sample rates, the average sequence length of processed audio clips is 559.

# Baseline and Reproducibility
We use non-autoregressive FastSpeech 2 and autoregressive Transformer-TTS as backbone networks. 

## Building Environment
```python
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

## Data Preparation
Download [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), create splits and generate audio manifests with
```shell
AUDIO_DATA_ROOT=<path>
AUDIO_MANIFEST_ROOT=<path>
NUMEXPR_MAX_THREADS=20 python -m examples.speech_synthesis.preprocessing.get_ljspeech_audio_manifest \
  --output-data-root ${AUDIO_DATA_ROOT} \
  --output-manifest-root ${AUDIO_MANIFEST_ROOT}
```

### Fastspeech 2 Spectrograms Extraction
Because FastSpeech 2 needs prediction of duration. We here provide two duration computation tools for duration extraction:

If using `g2pE` to compute durations, download [`g2pE`](https://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_mfa.zip) and set `TEXT_GRID_ZIP_PATH` to the path of `ljspeech_mfa.zip`.
```bash

AUDIO_MANIFEST_ROOT=<path>
FEATURE_MANIFEST_ROOT=<path>
TEXT_GRID_ZIP_PATH=<path>
NUMEXPR_MAX_THREADS=20 python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT} \
  --ipa-vocab --use-g2p --add-fastspeech-targets \
  --textgrid-zip ${TEXT_GRID_ZIP_PATH} 
```

If using `units` to compute durations, download [`units`](https://dl.fbaipublicfiles.com/fairseq/s2/ljspeech_hubert.tsv) and set `ID_TO_UNIT_TSV` to the path of `ljspeech_hubert.tsv`.
```bash

AUDIO_MANIFEST_ROOT=<path>
FEATURE_MANIFEST_ROOT=<path>
ID_TO_UNIT_TSV=<path>
NUMEXPR_MAX_THREADS=20 python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT} \
  --ipa-vocab --use-g2p --add-fastspeech-targets \
  --id-to-units-tsv ${ID_TO_UNIT_TSV}  
```

You can also generate durations by yourself using
a different software or model:
[Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get `g2pE` duration or 
[HuBERT](https://github.com/pytorch/fairseq/tree/main/examples/hubert) to get `units` duration.

**NOTE**: In our paper, results of FastSpeech 2 are produced by `g2pE` durations.

### Transformer-TTS Feature Extraction
```bash
python -m examples.speech_synthesis.preprocessing.get_feature_manifest \
  --audio-manifest-root ${AUDIO_MANIFEST_ROOT} \
  --output-root ${FEATURE_MANIFEST_ROOT} \
  --ipa-vocab --use-g2p &
```

## Training
We use 1×80GB A100 to train both FastSpeech 2 and Transformer TTS models.

#### Transformer TTS
```bash
fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-sentences 6 --max-update 200000 \
  --task text_to_speech --criterion tacotron2 --arch tts_transformer \
  --clip-norm 5.0 --n-frames-per-step 4 --bce-pos-weight 5.0 \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
  --encoder-normalize-before --decoder-normalize-before \
  --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss
```
where `SAVE_DIR` is the checkpoint root path. We set `--update-freq 8` to simulate 8 GPUs with 1 GPU. You may want to
update it accordingly when using more than 1 GPU.


#### FastSpeech 2
```bash
fairseq-train ${FEATURE_MANIFEST_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-sentences 6 --max-update 200000 \
  --task text_to_speech --criterion fastspeech2 --arch fastspeech2 \
  --clip-norm 5.0 --n-frames-per-step 1 \
  --dropout 0.1 --attention-dropout 0.1 \
  --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --seed 1 --update-freq 8 --eval-inference --best-checkpoint-metric mcd_loss
```

## Inference
Average the last 5 checkpoints, generate the test split spectrogram and waveform using the default Griffin-Lim vocoder:
```bash
SPLIT=test
CHECKPOINT_NAME=avg_last_5
CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_${CHECKPOINT_NAME}.pt
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 5 \
  --output ${CHECKPOINT_PATH}
```
**NOTE: you can just use the best checkpoint as our paper's setting. In this case, `CHECKPOINT_PATH=${SAVE_DIR}/checkpoint_best.pt`.**  
Then generate the waveforms to the `EVAL_OUTPUT_ROOT`:
```bash
EVAL_OUTPUT_ROOT=$SAVE_DIR/avg
python -m examples.speech_synthesis.generate_waveform ${FEATURE_MANIFEST_ROOT} \
  --config-yaml config.yaml --gen-subset ${SPLIT} --task text_to_speech \
  --path ${CHECKPOINT_PATH} --max-tokens 50000 --spec-bwd-max-iter 32 \
  --dump-waveforms --dump-target --results-path $EVAL_OUTPUT_ROOT
```

## Automatic Evaluation
We only use MCD/MSD metrics. You can also use other automatic metrics following the guidance of original files.

First generate the evaluation file:
```bash
python -m examples.speech_synthesis.evaluation.get_eval_manifest \
  --generation-root ${EVAL_OUTPUT_ROOT} \
  --audio-manifest ${AUDIO_MANIFEST_ROOT}/${SPLIT}.audio.tsv \
  --output-path ${EVAL_OUTPUT_ROOT}/eval.tsv \
  --vocoder griffin_lim --audio-format wav \
  --use-resynthesized-target 
```
#### MCD/MSD metric
```bash
python -m examples.speech_synthesis.evaluation.eval_sp \
  ${EVAL_OUTPUT_ROOT}/eval.tsv --mcd --msd
```
The numbers in `dist_per_syn_frm` column is the final results.
