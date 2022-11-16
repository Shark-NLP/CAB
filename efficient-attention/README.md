# Efficient Attention Library

## Overview
We provides a light-weight and plugin library for efficient attention. We show the supported attention mechanisms and their available attention patterns as follows: 

| Attention                                                                                                                                           | Noncausal Self               |   Causal Self           | Noncausal Cross              | Causal Cross |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------------|--------------------|--------------------|
| local attention [(Luong et al., 2015)](https://aclanthology.org/D15-1166.pdf)                                                                                                                                              | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| ABC [(Peng et al., 2021)](https://arxiv.org/pdf/2110.02488.pdf)                                                                                     | :heavy_check_mark: | :heavy_check_mark:                 | :heavy_check_mark: | :heavy_check_mark: |
| Nyströmformer [(Xiong et al., 2021)](https://arxiv.org/pdf/2102.03902.pdf)                                                                          | :heavy_check_mark: | :x:                | :x:                | :x:                |
| Performer [(Choromanski et al., 2020)](https://arxiv.org/pdf/2009.14794.pdf)  | :heavy_check_mark: | :x:                 |    :heavy_check_mark:             | :heavy_check_mark:                |
| LARA [(Zheng et al., 2022)](https://arxiv.org/pdf/2204.04667.pdf)                                                                                                                                               | :heavy_check_mark: | :x:                | :x:                | :x:                |
| cosFormer [(Qin et al., 2022)](https://arxiv.org/pdf/2202.08791.pdf) | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: |
| ProbSparse [(Zhou et al., 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17325) | :heavy_check_mark: | :x: | :x: | :x: |
| S4D [(Gu et al., 2022)](https://arxiv.org/abs/2206.11893) | :heavy_check_mark: | :heavy_check_mark:  | :x: | :x: | 
| LongShort [(Zhu et al., 2021)](https://arxiv.org/pdf/2107.02192.pdf) | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: |

**Note:** `S4D` and `LongShort` currently don't support [`fairseq incremental decoding`](https://fairseq.readthedocs.io/en/latest/models.html#incremental-decoding).

## Setup Details
To install efficient library in your environment, please use the following scripts to download the library:
```bash
# in efficient-attention directory
python -m pip install -e .
```




## Usage

### Import from library
To use some specific efficient attention module in your project, import them from `efficient_attention` library:
```python
import efficient_attention
# or
from efficient_attention import *
# or just some attention
from efficient_attention import LocalAttention
```
To use our library in `fairseq` or `paragen`, please make sure that your environment has installed `paragen` and `fairseq`. To use efficient attention (take `ABC` as an example) in `fairseq`, 
```python
# fairseq is in your pip list
from efficient_attention import ABC, fsmha 

@fsmha
class FSABC(ABC):
    pass
kwargs = dict(embed_dim=512, num_heads=8, dropout=0.2)
attn = FSABC(**kwargs)
```


To use efficient attention in `paragen`,
```bash
cd Paragen/examples/summarization
# assuming in the working directory
cp -rf <path/to/Long-Sequence-Benchmark/efficient-attention/efficient_attention/plugins/efficient_transformers> ./
paragen-run --config configs/train-bart-base.yaml --lib efficient_transformers
```


### Initialization
We provide two arguments ``cross`` and ``causal`` for each attention to support four attention patterns.
For example, if I want to initialize a noncausal self `ABC` instance, specify like this:
```python
from efficient_attention import ABC
kwargs = dict(embed_dim=512, num_heads=8, dropout=0.3)
attn = ABC(cross=False, causal=False, **kwargs)
```
If I want to initialize a causal cross `ABC` instance, initialize it like this:
```python
from efficient_attention import ABC
kwargs = dict(embed_dim=512, num_heads=8, dropout=0.3)
attn = ABC(cross=True, causal=True, **kwargs)
```
If you initialize an efficient attention with its non-supportive attention pattern, it will raise an `Assertion Error`:
```python
from efficient_attention import ProbSparse
kwargs = dict(embed_dim=512, num_heads=8, dropout=0.3)
attn = ProbSparse(cross=True, causal=False, **kwargs)
# AssertionError: ProbSparse cannot do cross attention now
```
For more initialization help, you can use your `IPython interpreter` and type the following commands to see how to use each efficient attention
```python
In [1]: from efficient_attention import LocalAttention
In [2]: LocalAttention?
Init signature: LocalAttention(wsize=15, **kwargs)
Docstring:
Usage:

from efficient_attention import LocalAttention
attn = LocalAttention(embed_dim=embed_dim, num_heads=num_heads,wsize=wsize,causal=is_causal, dropout=dropout)

result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask, incremental_state=incremental_state)
Init docstring: Initializes internal Module state, shared by both nn.Module and ScriptModule.
File:           <path/to/your/installation>
Type:           type
Subclasses:
```

For `fsmha` wrapped efficient attention, they accept the same parameters as their unwrapped version.
For `EfficientTransformerEncoder`, again type `EfficientTransformerEncoder?` for detailed description in `IPython` interpreter.

### Forward pass
For execution of typical attention, we unify our forward interface with `fairseq`. 
Specifically, our forward method has the following arguments:
```python
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        need_head_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        static_kv: bool = False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        batch_first: bool = False,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
        Args:
            query : Query embeddings of shape `(L_q, B, D)`
            key   : Key embeddings of shape `(L_k, B, D)`
            value : Value embeddings of shape `(L_k, B, D)`
            query_padding_mask: If specified, a mask of shape `(B, L_q)` indicating which elements within ``query``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            key_padding_mask: If specified, a mask of shape `(B, L_k)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``. Defaults to True.
            need_head_weights: return the attention weights for each head. 
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                       `(L_q, L_k)` or `(B\cdot\text{num\_heads}, L_q, L_k)`
            static_kv: If specified, key and value are computed only once and cached for future computation. Defaults to False.
            incremental_state: If specified, it caches historical internal states and is further updated after current computation process. Defaults to None.
            batch_first: Whether to transform shape so that each tensor's shape is (B, ...). Defaults to False.
        """
```
For more details of each argument, again use `IPython interpreter` to see the usage.
```python
In [1]: from efficient_attention import LocalAttention
In [2]: LocalAttention.forward?
```
