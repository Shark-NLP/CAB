# This file is the implementation of Nystromformer introduced in 
# "Nystromformer: A Nystrom-based Algorithm for Approximating Self-Attention"
# Supported pattern of the code: Noncausal Self.
# The code comes from https://github.com/mlpen/Nystromformer.

import math
from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from efficient_attention import MultiheadAttention, register_cls, add_nested_argument


@register_cls
class NystromAttention(MultiheadAttention):
    r"""
    Usage:
    
    from efficient_attention import NystromAttention
    attn = NystromAttention(embed_dim=embed_dim, num_heads=num_heads,num_landmarks=num_landmarks, dropout=dropout, conv_kernel_size=conv_kernel_size)

    result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask)
    
    """
    
    def __init__(self, num_landmarks=16, conv_kernel_size=None, **kwargs):
        super().__init__(**kwargs)
        assert self.causal == False, f"{self.name} cannot do causal attention now"
        assert self.cross == False, f"{self.name} cannot do cross attention now"

        self.num_landmarks = num_landmarks

        self.conv = None
        if conv_kernel_size is not None:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads, out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1), padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads)

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bsz: int,
        attn_mask: Optional[Tensor] = None,
        query_padding_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Computes Nystrom attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q (Tensor): query tensors. :math:`(B, Nt, E)` where B is batch size, Nt is the sequence length of query,
                and E is embedding dimension.
            k (Tensor): key tensors. :math:`(B, Ns, E)` where B is batch size, Nt is the sequence length of key,
                and E is embedding dimension.
            v (Tensor): value tensors. :math:`(B, Ns, E)` where B is batch size, Nt is the sequence length of value,
                and E is embedding dimension.
            bsz (int): batch size
            attn_mask (Optional[Tensor], optional): optional tensor containing mask values to be added to calculated
                attention; either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`. Defaults to None.
            query_padding_mask (Optional[Tensor], optional):  If specified, a mask of shape :math:`(B, Nt)` indicating 
                which elements within ``query`` to ignore for the purpose of attention (i.e. treat as "padding"). 
                Binary and byte masks are supported. For a binary mask, a ``True`` value indicates that the corresponding 
                ``query`` value will be ignored for the purpose of attention. For a byte mask, a non-zero value 
                Indicates that the corresponding ``query`` value will be ignored. Defaults to None.
            key_padding_mask (Optional[Tensor], optional): If specified, a mask of shape :math:`(B, NS)` indicating 
                which elements within ``key`` to ignore for the purpose of attention (i.e. treat as "padding"). 
                Binary and byte masks are supported. For a binary mask, a ``True`` value indicates that the corresponding 
                ``key`` value will be ignored for the purpose of attention. For a byte mask, a non-zero value 
                Indicates that the corresponding ``key`` value will be ignored. Defaults to None.
            incremental_state (Optional[Dict[str, Dict[str, Optional[Tensor]]]], optional): If specified, it caches historical 
                internal key, value and key_padding_mask states: saved_state=incremental_state[self.name], and saved_state 
                has three components: ``prev_key`` :math: `(B, N_{<=i}, E)`, ``prev_value`` :math: `(B, N_{<=i}, E)`, and 
                ``prev_key_padding_mask` :math: `(B, N_{<=i})`. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        if attn_mask is not None:
            warnings.warn("`attn_mask` arguments make no sense in `NystromAttention`")
        # assert attn_mask is None, 'causal attention is not supported now!'
        B, N, D = q.shape  # [B * num_heads, N, D]
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1).repeat(self.num_heads, 1, 1).type_as(v)
            v = v * (1. - mask)
            k = k * (1. - mask)
        q = q / math.sqrt(self.head_dim)
        k = k / math.sqrt(self.head_dim)
        segs = N // self.num_landmarks
        if N <= self.num_landmarks:
            attn_logits = q @ k.transpose(-1, -2)
            if key_padding_mask is not None:
                attn_logits = attn_logits.masked_fill(
                    key_padding_mask.unsqueeze(-2).repeat(self.num_heads, 1, 1).to(torch.bool),
                    float("-inf"))
            attn = F.softmax(attn_logits, dim=-1)
            X = torch.matmul(attn, v)
        else:
            if (N % self.num_landmarks == 0):
                Q_landmarks = q.reshape(B, self.num_landmarks, N // self.num_landmarks, D).mean(dim=-2)
                K_landmarks = k.reshape(B, self.num_landmarks, N // self.num_landmarks, D).mean(dim=-2)
            else:
                num_k = (segs + 1) * self.num_landmarks - N

                keys_landmarks_f = k[:, :num_k * segs, :].reshape(
                    B, num_k, segs, D).mean(dim=-2)
                keys_landmarks_l = k[:, num_k * segs:, :].reshape(
                    B, self.num_landmarks - num_k, segs + 1, D).mean(dim=-2)
                K_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim=-2)

                queries_landmarks_f = q[:, :num_k * segs, :].reshape(
                    B, num_k, segs, D).mean(dim=-2)
                queries_landmarks_l = q[:, num_k * segs:, :].reshape(
                    B, self.num_landmarks - num_k, segs + 1, D).mean(dim=-2)
                Q_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim=-2)

            kernel_1 = F.softmax(torch.matmul(q, K_landmarks.transpose(-1, -2)), dim=-1)
            kernel_2 = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim=-1)
            kernel_3_logits = torch.matmul(Q_landmarks, k.transpose(-1, -2))
            if key_padding_mask is not None:
                kernel_3_logits = kernel_3_logits.masked_fill(
                        key_padding_mask.unsqueeze(-2).repeat(self.num_heads, 1, 1).to(torch.bool),
                        float("-inf")
                    )
            kernel_3 = F.softmax(kernel_3_logits, dim=-1)

            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, v))

        if self.conv is not None:
            conv_v = v.reshape(bsz, self.num_heads, -1, self.head_dim)
            conv_output = self.conv(conv_v)
            X += conv_output.reshape(bsz * self.num_heads, -1, self.head_dim)
            # X += self.conv(v)
            # if key_padding_mask is not None:
            #     X += self.conv(v * (1. - key_padding_mask.float().unsqueeze(-1).repeat(self.num_heads, 1, 1)))
            # else:
            #     X += self.conv(v)
        return X, None

    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype)
        K = mat
        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        # This original implementation is more conservative to compute coefficient of Z_0.
        V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V
    
    @staticmethod
    def add_attn_specific_args(parent_parser):
        if hasattr(super(NystromAttention, NystromAttention), "add_attn_specific_args"):
            parent_parser = super(NystromAttention, NystromAttention).add_attn_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Attention")
        add_nested_argument(parser, '--conv-kernel-size', default=None, type=int,
                                help='number of random features')
        add_nested_argument(parser, '--num-landmarks', default=16, type=int,
                                help='number of random features')
        return parent_parser
