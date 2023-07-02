# This file is the implementation of FlashAttention introduced in
# "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
# Supported pattern of the code: Noncausal Self, Causal Self.
# The code comes from https://github.com/HazyResearch/flash-attention.git.

import math
from typing import Dict, Optional, Tuple

import torch
# import flash_attn_cuda
import torch.nn as nn
import torch.nn.functional as F
from efficient_attention import AbstractAttention, register_cls
from efficient_attention.modules.multihead_attention import \
    _append_prev_key_padding_mask
from einops import rearrange, repeat
from torch import Tensor
from flash_attn.modules.mha import FlashSelfAttention, FlashCrossAttention


class FlashAttention(AbstractAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64, 128], "Only support head_dim == 16, 32, 64, or 128"
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        self.inner_attn = FlashSelfAttention(causal=self.causal, attention_dropout=dropout, softmax_scale=self.head_dim ** -0.5)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.k_proj, self.v_proj, self.q_proj, self.out_proj]:
            kwargs = {'gain': 1 / math.sqrt(2)} if proj is not self.out_proj else {}
            nn.init.xavier_uniform_(proj.weight, **kwargs)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)

    def forward(self, query,
                key=None,
                value=None,
                query_padding_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = False,
                need_head_weights: bool = False,
                attn_mask: Optional[Tensor] = None,
                static_kv: bool = False,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                batch_first: bool = False,
                **kwargs
                ) -> Tuple[Tensor, Optional[Tensor]]:
        x = query
        if not batch_first:
            x = torch.transpose(x, 0, 1)

        qkv = self.Wqkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        context = self.inner_attn(qkv)
        context = rearrange(context, 'b s h d -> b s (h d)')

        attn = self.out_proj(context.transpose(1, 2)).transpose(1, 2)

        if not batch_first:
            attn = attn.transpose(0, 1)
        return attn, None
