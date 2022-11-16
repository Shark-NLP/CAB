# Attention Pattern Validation

We provide [`test_available_pattern_for_attention.py`](./attention_pattern/test_available_pattern_for_attention.py) to test whether an attention mechanism can perform four attention patterns. It may not be suitable for chunk-based attention mechanism, such as `Long-Short Transformer`.
```python
python ./attention_pattern/test_available_pattern_for_attention.py
```

# Efficiency Length
We take `ABC` as an example to compute the efficiency length. You should first add attention class as follows:
```python
def build_self_attn(attention_type, max_seq_len=8192, causal=False):
    if attention_type == 'vanilla':
        return MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
        )
    elif attention_type == 'abc':
        return ABC(
            embed_dim=embed_dim,
            num_landmarks=landmarks,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal
        )
    return None
```
Then execute the following command:
```shell
export attention_type=abc
bash ./efficiency_length/get_efficiency_length.sh ${attention_type}
```

# Compositional Index
We provide [`compositional_index.py`](./compositional_index/compositional_index.py) to calculate `CI` for each attention pattern. As a demonstration, `example.txt` contains vanilla's scores. 
```shell
export score_file=compositional_index/example.txt # The file contains scores of an attention on tasks
python ./compositional_index/compositional_index.py ${score_file}
```
The `example.txt` should contain 4 lines **(noncausal self, causal self, noncausal cross, causal cross)** as follows:

TTS <br> FastSpeech 2  <br> MCD | TTS <br> FastSpeech 2  <br> MSD | TTS <br> Transformer-TTS <br> MCD | TTS <br> Transformer-TTS <br> MSD | Sum <br> R-1 | Sum <br> R-2 | Sum <br> R-L | SR <br> PSNR | SR <br> SSMI | MLM <br> PPL
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- |


TTS <br> Transformer-TTS <br> MCD | TTS <br> Transformer-TTS <br> MSD | Sum <br> R-1 | Sum <br> R-2 | Sum <br> R-L | LM <br> PPL
--- | --- | --- | --- | --- | --- |

PCC <br> CD-l1 | PCC <br> CD-l2 | PCC <br> F-Score | LSTF-ETT <br> MSE | LSTF-ETT <br> MAE | LSTF-ECL <br> MSE | LSTF-ECL <br> MAE | LSTF-Weather <br> MSE | LSTF-Weather <br> MAE |
--- | --- | --- | --- | --- | --- | --- | --- | --- |

TTS <br> Transformer-TTS <br> MCD | TTS <br> Transformer-TTS <br> MSD | Sum <br> R-1 | Sum <br> R-2 | Sum <br> R-L 
--- | --- | --- | --- | --- 
