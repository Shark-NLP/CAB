# This file is the implementation of ABC introduced in 
# "ABC: Attention with Bounded-Memory Control"
# Supported pattern of the code: Noncausal Self, Causal Self, Noncausal Cross, Causal Cross.
from typing import Optional, Tuple, Dict
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from efficient_attention import MultiheadAttention, register_cls, add_nested_argument


@register_cls
class ABC(MultiheadAttention):
    r"""
    Usage:
    
    from efficient_attention import ABC
    attn = ABC(embed_dim=embed_dim, num_heads=num_heads,num_landmarks=num_landmarks, dropout=dropout, causal=is_causal)

    result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask)
    
    """
    
    def __init__(self,
                 num_landmarks=16,
                 **kwargs):
        super(ABC, self).__init__(**kwargs)
        self.num_landmarks = num_landmarks

        self.compress_k = nn.Parameter(torch.randn(self.head_dim, num_landmarks)) 
        self.compress_v = nn.Parameter(torch.randn(self.head_dim, num_landmarks)) 
        nn.init.xavier_uniform_(self.compress_k, gain=2 ** -.5)
        nn.init.xavier_uniform_(self.compress_v, gain=2 ** -.5)

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
        Computes attention with bounded memory on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q (Tensor): query tensors. :math:`(B, Nt, E)` where B is batch size, Nt is the sequence length of query,
                and E is embedding dimension.
            k (Tensor): key tensors. :math:`(B, Ns, E)` where B is batch size, Nt is the sequence of key,
                and E is embedding dimension.
            v (Tensor): value tensors. :math:`(B, Ns, E)` where B is batch size, Nt is the sequence of value,
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
            warnings.warn("`attn_mask` arguments make no sense in `ABC`")
        bsz, N, D = k.shape


        q = q * self.head_dim ** -0.5
        if not self.causal or self.cross:
            if key_padding_mask is not None:
                mask = key_padding_mask[:, None].repeat(1, self.num_heads, 1).view(bsz, 1, N).to(torch.bool)
            k_logits = torch.einsum("bnd,dr->bnr", k, self.compress_k).transpose(-1, -2).contiguous()
            if key_padding_mask is not None:
                k_logits = k_logits.masked_fill(mask, float('-inf'))
            k_probs = torch.softmax(k_logits, dim=-1)
            k = torch.matmul(k_probs, k)  # [b*h, c, d]

            attn_probs = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)

            v_logits = torch.einsum("bnd,dr->bnr", v, self.compress_v).transpose(-1, -2).contiguous()
            if key_padding_mask is not None:
                v_logits = v_logits.masked_fill(mask, float('-inf'))
            v_probs = torch.softmax(v_logits, dim=-1)
            v = torch.matmul(v_probs, v)  # [b*h, c, d]

            output = torch.matmul(attn_probs, v)
        else:
            if key_padding_mask is not None:
                mask = key_padding_mask[:, None].repeat(1, self.num_heads, 1).view(bsz, N, 1).to(torch.bool)
            alpha_k = torch.einsum("bnd,dr->bnr", k, self.compress_k)
            if key_padding_mask is not None:
                alpha_k = alpha_k.masked_fill(mask, float('-inf'))
            alpha_v = torch.einsum("bnd,dr->bnr", v, self.compress_v)
            if key_padding_mask is not None:
                alpha_v = alpha_v.masked_fill(mask, float('-inf'))

            if incremental_state is None:
                alpha_k_cumsum = torch.logcumsumexp(alpha_k, 1)
                k_logits = torch.exp(alpha_k - alpha_k_cumsum)
                k_logits = torch.einsum("bnr,bnd->bnrd", k_logits, k)

                
                alpha_v_cumsum = torch.logcumsumexp(alpha_v, 1)
                v_logits = torch.exp(alpha_v - alpha_v_cumsum)
                v_logits = torch.einsum("bnr,bnd->bnrd", v_logits, v)
                
                attn_probs = torch.softmax(torch.einsum("bnd, bnrd->bnr", q, k_logits), -1)
                output = torch.einsum("bnr, bnrd->bnd", attn_probs, v_logits)
            else:
                alpha_k_cumsum = torch.logcumsumexp(alpha_k, 1)
                k_logits = torch.exp(alpha_k - alpha_k_cumsum)
                k_logits = torch.einsum("br,bd->brd", k_logits[:, -1], k[:, -1])

                
                alpha_v_cumsum = torch.logcumsumexp(alpha_v, 1)
                v_logits = torch.exp(alpha_v - alpha_v_cumsum)
                v_logits = torch.einsum("br,bd->brd", v_logits[:, -1], v[:, -1])
                
                attn_probs = torch.softmax(torch.einsum("bnd,brd->bnr", q, k_logits), -1)
                output = torch.einsum("bnr, brd->bnd", attn_probs, v_logits)

            
        return output, None

    @staticmethod
    def add_attn_specific_args(parent_parser):
        if hasattr(super(ABC, ABC), "add_attn_specific_args"):
            parent_parser = super(ABC, ABC).add_attn_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Attention")

        add_nested_argument(parser, '--num-landmarks', default=16, type=int,
                                help='number of random features')
        add_nested_argument(parser, '--causal', default=False, action='store_true')
        return parent_parser
