# This file is the implementation of local attention introduced in 
# "Effective Approaches to Attention-based Neural Machine Translation"
# Supported pattern of the code: Noncausal Self, Causal Self.
import math
from typing import Optional, Tuple, Dict
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from efficient_attention import MultiheadAttention, register_cls, add_nested_argument
from efficient_attention.modules.multihead_attention import _append_prev_key_padding_mask


@register_cls
class LocalAttention(MultiheadAttention):
    r"""
    Usage:
    
    from efficient_attention import LocalAttention
    attn = LocalAttention(embed_dim=embed_dim, num_heads=num_heads,wsize=wsize,causal=is_causal, dropout=dropout)

    result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask, incremental_state=incremental_state)
    
    """
    def __init__(self,
                 wsize=15,
                 **kwargs):
        super(LocalAttention, self).__init__(**kwargs)

        assert self.cross == False, f"{self.name.split('.')[0]} cannot do cross attention now"

        self.wsize = wsize

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
        Computes local attention on query, key and value tensors, using
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
        if incremental_state is None:
            k, v, key_padding_mask = self._reshape_kv(k, v, key_padding_mask) # (B, Nt, wsize, E), (B, Nt, wsize)
        # print(q.shape, k.shape, key_padding_mask if key_padding_mask is None else key_padding_mask.shape)
        if key_padding_mask is not None:

            if len(key_padding_mask.shape) == 2:
                # add num_heads to the first and Nt information
                if key_padding_mask.shape[0] == q.shape[1]:
                    key_padding_mask = key_padding_mask.unsqueeze(0).repeat(q.shape[0], 1, 1)
                else:
                    key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1).reshape(bsz * self.num_heads, -1, k.shape[-2])

            assert tuple(key_padding_mask.shape) == (bsz * self.num_heads, q.shape[1], k.shape[-2])
        q = q / math.sqrt(self.head_dim)
        # (B, Nt, E) x (B, Nt, wsize, E) -> (B, Nt, wsize)
        attn_weights = torch.einsum('bnd,bnwd->bnw', q, k)

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.to(torch.bool),
                float("-inf"),
            )
        # print(attn_weights)
        attn_weights_float = F.softmax(
            attn_weights, dim=-1,
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn = torch.einsum('bnw,bnwd->bnd', attn_probs, v)
        return attn, attn_weights

    def _update_saved_states(
        self,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor,
        saved_state: Dict[str, Optional[Tensor]],
        bsz: int,
        static_kv: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                k = torch.cat([prev_key, k], dim=1)[:, -self.wsize:]
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                v = torch.cat([prev_value, v], dim=1)[:, -self.wsize:]
        prev_key_padding_mask: Optional[Tensor] = None
        if "prev_key_padding_mask" in saved_state:
            prev_key_padding_mask = saved_state["prev_key_padding_mask"]
        key_padding_mask = _append_prev_key_padding_mask(
            key_padding_mask=key_padding_mask,
            prev_key_padding_mask=prev_key_padding_mask,
            batch_size=bsz,
            src_len=k.shape[1],
            static_kv=static_kv,
        )
        if not static_kv and key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, -self.wsize:]

        k = k.view(bsz * self.num_heads, 1, -1, self.head_dim)
        v = v.view(bsz * self.num_heads, 1, -1, self.head_dim)

        saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_key_padding_mask"] = key_padding_mask

        return k, v, key_padding_mask

    def _reshape_kv(self, k, v, key_padding_mask=None):
        bs, klen, d = k.shape  # klen = srclen
        if self.causal:
            shift = torch.arange(-self.wsize+1, 1, device=k.device)
            wsize = self.wsize 
        else:
            shift = torch.arange(-self.wsize, self.wsize+1, device=k.device)
            wsize = self.wsize * 2 + 1
        shift = shift.unsqueeze(0)
        idx = torch.arange(0, klen, device=k.device).unsqueeze(-1)
        idx = idx + shift
        mask = (idx < 0) | (idx >= klen)

        idx = idx.clamp(0, k.shape[1] - 1)
        idx_b = idx.view(-1).unsqueeze(0).repeat(bs, 1)  # (bs, srclen x wsize)
        idx_bd = idx_b.unsqueeze(-1).repeat(1, 1, d)  # (bs, srclen x wsize, dim)
        k = k.gather(1, idx_bd).reshape(bs, klen, wsize, d)
        v = v.gather(1, idx_bd).reshape(bs, klen, wsize, d)
        if key_padding_mask is not None:
            # kye padding mask: (bs, klen / numheads)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1).view(bs, klen)
            key_padding_mask = key_padding_mask.gather(1, idx_b).reshape(bs, klen, wsize)
            mask = mask.unsqueeze(0) & key_padding_mask

        # mask: (bs, klen, wsize)
        return k, v, mask
    
    @staticmethod
    def add_attn_specific_args(parent_parser):
        if hasattr(super(LocalAttention, LocalAttention), "add_attn_specific_args"):
            parent_parser = super(LocalAttention, LocalAttention).add_attn_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Attention")

        add_nested_argument(parser, '--wsize', default=15, type=int,
                                help='window size')
        add_nested_argument(parser, '--causal', default=False, action='store_true')
        return parent_parser
