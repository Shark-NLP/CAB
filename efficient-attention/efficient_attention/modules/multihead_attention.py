# This file is the implementation of Vanilla Attention introduced in 
# "Attention is All You Need"
# Supported pattern of the code: Noncausal Self, Causal Self, Noncausal Cross, Causal Cross.

import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from efficient_attention import AbstractAttention, register_cls


@register_cls
class MultiheadAttention(AbstractAttention):
    r"""

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        
    Usage:
    
        from efficient_attention import MultiheadAttention
        attn = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask, incremental_state=incremental_state)
    
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        **kwargs
    ) -> None:
        super(MultiheadAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in [self.k_proj, self.v_proj, self.q_proj, self.out_proj]:
            kwargs = {'gain': 1 / math.sqrt(2)} if self.qkv_same_dim and proj is not self.out_proj else {}
            nn.init.xavier_uniform_(proj.weight, **kwargs)
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

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
        r"""Forward attention layer.

        Args:
            query (Tensor): Query embeddings of shape :math:`(Nt, B, E_q)` when ``batch_first=False`` or :math:`(B, Nt, E_q)`
                when ``batch_first=True``, where :math:`Nt` is the sequence length of query, :math:`B` is the batch size,
                and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
                key-value pairs to produce the output. See "Attention Is All You Need" for more details.
            key (Optional[Tensor], optional): Key embeddings of shape :math:`(Ns, B, E_k)` when ``batch_first=False`` or :math:`(B, Ns, E_k)` when
                ``batch_first=True``, where :math:`Ns` is the sequence length of key, :math:`B` is the batch size, and
                :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details. Defaults to None.
            value (Optional[Tensor], optional): Value embeddings of shape :math:`(Ns, B, E_v)` when ``batch_first=False`` or :math:`(B, Ns, E_v)` when
                ``batch_first=True``, where :math:`Ns` is the sequence length of value, :math:`B` is the batch size, and
                :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details. Defaults to None.
            query_padding_mask (Optional[Tensor], optional): If specified, a mask of shape :math:`(B, Nt)` indicating which elements within ``query``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``query`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``query``
                value will be ignored. Defaults to None.
            key_padding_mask (Optional[Tensor], optional): If specified, a mask of shape :math:`(B, Ns)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                value will be ignored. Defaults to None.
            need_weights (bool, optional): If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``. Defaults to True.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. If specified, it defaults to
                return the average attention weights over all heads. Defaults to False.
            attn_mask (Optional[Tensor], optional): If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(Nt, Ns)` or :math:`(B\cdot\text{num\_heads}, Nt, Ns)`, where :math:`B` is the batch size,
                :math:`Nt` is the target sequence length, and :math:`Ns` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight. Defaults to None.
            static_kv (bool, optional): If specified, key and value are computed only once and cached for future computation. Defaults to False.
            incremental_state (Optional[Dict[str, Dict[str, Optional[Tensor]]]], optional): If specified, it caches historical internal states
                and is further updated after current computation process. Defaults to None.
            batch_first (bool, optional): Whether to transform shape so that each tensor's shape is (B, ...). Defaults to False.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: 
                - **attn_output** - Attention outputs of shape :math:`(Nt, B, E)` when ``batch_first=False`` or
                    :math:`(B, Nt, E)` when ``batch_first=True``, where :math:`Nt` is the target sequence length, :math:`B` is
                    the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
                - **attn_output_weights** - Attention output weights of shape :math:`(B, Nt, Ns)`, where :math:`B` is the batch
                    size, :math:`Nt` is the target sequence length, and :math:`Ns` is the source sequence length. Only returned
                    when ``need_weights=True``.
        """
            
        if need_head_weights:
            need_weights = True

        if key is None:
            key = query
        if value is None:
            value = key

        # set up shape vars
        if batch_first:
            bsz, tgt_len, embed_dim = query.shape
        else:
            tgt_len, bsz, embed_dim = query.shape

        if incremental_state is not None:
            saved_state: Optional[Dict[str, Optional[Tensor]]] = {}
            key, value = self._get_saved_states(incremental_state, saved_state, static_kv, key, value)
        else:
            saved_state: Optional[Dict[str, Optional[Tensor]]] = None

        # check shape of input tensor
        self._input_shape_check(key, value, key_padding_mask, batch_first)

        q, k, v = self._in_proj(query, key, value)

        # prep attention mask
        attn_mask, key_padding_mask = _prep_mask(attn_mask, key_padding_mask)

        # add bias along batch dimension (currently second)
        if self.bias_k is not None and self.bias_v is not None:
            k, v, attn_mask, key_padding_mask = self._add_bias(k, v, attn_mask, key_padding_mask)
        else:
            assert self.bias_k is None and self.bias_v is None

        # reshape q, k, v for multihead attention and make em batch first
        q, k, v = self._reshape_qkv(q, k, v, batch_first)

        if saved_state is not None:
            k, v, key_padding_mask = self._update_saved_states(k, v, key_padding_mask, saved_state, bsz, static_kv)
            incremental_state[self.name] = saved_state

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            k, v, key_padding_mask, attn_mask = self._pad_zero_attn(k, v, key_padding_mask, attn_mask, bsz)

        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = self._apply_attention(q, k, v, bsz, attn_mask, query_padding_mask, key_padding_mask, incremental_state)
        if batch_first:
            attn_output = attn_output.reshape(bsz, self.num_heads, -1, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, -1)
        else:
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        attn_weights = None
        if need_weights and attn_output_weights is not None:
            attn_weights = self.calc_weight(attn_output_weights, need_head_weights)

        return attn_output, attn_weights

    def _input_shape_check(
        self,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        batch_first: bool = False,
    ):
        if key is not None and value is not None:
            assert key.shape[-1] == value.shape[-1] == self.embed_dim
            assert key.shape[0] == value.shape[0] and key.shape[1] == value.shape[1]
            if key_padding_mask is not None:
                if batch_first:
                    assert key.shape[0] == key_padding_mask.shape[0] and key.shape[1] == key_padding_mask.shape[1]
                else:
                    assert key.shape[0] == key_padding_mask.shape[1] and key.shape[1] == key_padding_mask.shape[0]

    def _in_proj(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # compute in-projection
        q = self.q_proj(query) if query is not None else None
        k = self.k_proj(key) if key is not None else None
        v = self.v_proj(value) if value is not None else None
        return q, k, v

    def _add_bias(
        self,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor,
        key_padding_mask: Tensor,
        batch_first: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        bsz = key_padding_mask.shape[0]
        pad_k = self.bias_k.repeat(bsz, 1, 1) if batch_first else self.bias_k.repeat(1, bsz, 1)
        pad_v = self.bias_v.repeat(bsz, 1, 1) if batch_first else self.bias_v.repeat(1, bsz, 1)
        k = torch.cat([k, pad_k])
        v = torch.cat([v, pad_v])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        return k, v, attn_mask, key_padding_mask

    def _reshape_qkv(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        batch_first: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if batch_first:
            if q is not None:
                q = q.reshape(q.shape[0], q.shape[1], -1, self.head_dim).transpose(1, 2)
            if k is not None:
                k = k.reshape(k.shape[0], k.shape[1], -1, self.head_dim).transpose(1, 2)
            if v is not None:
                v = v.reshape(v.shape[0], v.shape[1], -1, self.head_dim).transpose(1, 2)
            return q.reshape(-1, q.shape[2], self.head_dim), \
                k.reshape(-1, k.shape[2], self.head_dim), \
                    v.reshape(-1, v.shape[2], self.head_dim)
        if q is not None:
            q = q.contiguous().view(q.shape[0], -1, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(k.shape[0], -1, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(v.shape[0], -1, self.head_dim).transpose(0, 1)
        return q, k, v

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
    ) -> Tuple[Tensor, Tensor]:
        r"""Computes scaled dot product attention on query, key and value tensors, using
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
        tgtlen, srclen = q.shape[1], k.shape[1]
        q = q / math.sqrt(self.head_dim)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgtlen, srclen)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgtlen, srclen)

        attn_weights_float = F.softmax(
            attn_weights, dim=-1,
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_probs, v)
        return attn, attn_weights

    def _pad_zero_attn(
        self,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Tensor,
        attn_mask: Tensor,
        bsz: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert bsz == key_padding_mask.shape[0]
        zero_attn_shape = (bsz * self.num_heads, 1, self.head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        return k, v, key_padding_mask, attn_mask

    def _get_saved_states(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        saved_state: Dict[str, Optional[Tensor]],
        static_kv: bool,
        key: Tensor,
        value: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.name in incremental_state:
            for k, v in incremental_state[self.name].items():
                saved_state[k] = v
            if static_kv:
                key = value = None
        return key, value

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
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                v = torch.cat([prev_value, v], dim=1)
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

        saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_key_padding_mask"] = key_padding_mask

        return k, v, key_padding_mask

    def calc_weight(self,
                    attn_output_weights: Tensor,
                    need_head_weights: bool):
        bsz_hn, tgt_len, src_len = attn_output_weights.shape
        attn_output_weights = attn_output_weights.view(
            bsz_hn // self.num_heads, self.num_heads, tgt_len, src_len
        ).transpose(0, 1)
        if not need_head_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.mean(dim=0)
        return attn_output_weights
    
    @staticmethod
    def add_attn_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Attention")
        return parent_parser


def _prep_mask(
    attn_mask: Optional[Tensor],
    key_padding_mask: Optional[Tensor]
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. "
                          "Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)

    if key_padding_mask is not None:
        if key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. "
                          "Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)
    return attn_mask, key_padding_mask


def _append_prev_key_padding_mask(
    key_padding_mask: Optional[Tensor],
    prev_key_padding_mask: Optional[Tensor],
    batch_size: int,
    src_len: int,
    static_kv: bool,
) -> Optional[Tensor]:
    # saved key padding masks have shape (bsz, seq_len)
    if prev_key_padding_mask is not None and static_kv:
        new_key_padding_mask = prev_key_padding_mask
    elif prev_key_padding_mask is not None and key_padding_mask is not None:
        new_key_padding_mask = torch.cat(
            [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
        )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
    elif prev_key_padding_mask is not None:
        if src_len > prev_key_padding_mask.shape[1]:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.shape[1]),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask.float()
    elif key_padding_mask is not None:
        if src_len > key_padding_mask.shape[1]:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.shape[1]),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = key_padding_mask.float()
    else:
        new_key_padding_mask = prev_key_padding_mask
    return new_key_padding_mask
