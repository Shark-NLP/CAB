# This file is the implementation of Cosformer introduced in 
# "COSFORMER : RETHINKING SOFTMAX IN ATTENTION"
# Supported pattern of the code: Noncausal Self, Noncausal Cross.
# The code comes from https://github.com/OpenNLPLab/cosFormer.
from typing import Dict, Optional, Tuple
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from efficient_attention import AbstractAttention, register_cls, add_nested_argument


@register_cls
class CosformerAttention(AbstractAttention):
    """
    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        d_key: Total number of features for keys. Default: ``None``
        d_values: Total number of features for values. Default: ``None``
        has_outproj: Linearly transform the outputs of the attention mechanism
        act_fun: Type of activation function
    
    Usage:
    from efficient_attention import CosformerAttention
    attn = CosformerAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, causal=is_causal)

    result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask, incremental_state=incremental_state)
    
    """
    
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        has_outproj=True,
        act_fun="relu",
        **kwargs
    ):
        super(CosformerAttention, self).__init__(**kwargs)

        assert self.causal == False, f"{self.name.split('.')[0]} cannot do causal attention now"

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if kdim is not None else embed_dim
        self.num_heads = num_heads
        self.has_outproj = has_outproj
        self.act_fun = self.get_act_fun(act_fun)
        # q, k, v projection
        self.k_proj = nn.Linear(self.kdim, embed_dim)
        self.v_proj = nn.Linear(self.vdim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # outprojection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # dropout rate
        self.dropout_rate = dropout

        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu
    
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
        static_kv: bool= False,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        batch_first: bool = False,
        eps: Optional[float] = 1e-6,
        **kwargs
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            attn_mask (Optional[Tensor], optional): typically used to implement causal attention, 
            where the mask prevents the attention from looking forward in time (default: None).
        """
        if attn_mask is not None:
            warnings.warn("`attn_mask` arguments make no sense in `CosformerAttention`")
        if key == None:
            key = query
        if value == None:
            value = query
        
        num_heads = self.num_heads
        if batch_first:
            bsz, tgt_len, embed_dim = query.size()
            src_len = key.size(1)
        else:
            tgt_len, bsz, embed_dim = query.size()
            src_len = key.size(0)
        
        # if incremental_state is not None:
        #     saved_state: Optional[Dict[str, Optional[Tensor]]] = {}
        #     key, value = self._get_saved_states(incremental_state, saved_state, static_kv, key, value)
        # else:
        #     saved_state: Optional[Dict[str, Optional[Tensor]]] = None

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q, k, v = self._reshape_qkv(q, k, v, batch_first)
        # if saved_state is not None:
        #     k, v, key_padding_mask = self._update_saved_states(k, v, key_padding_mask, saved_state, bsz, static_kv)
        #     incremental_state[self.name] = saved_state
        # q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # # (N * h, S, d)
        # k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # # (N * h, S, d)
        # v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        if self.causal:
            ## Need to improve speed!
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
            kv_ = torch.einsum("nld,nlm->nldm", k_, v)
            # if self.training:
            if incremental_state is None:
            # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
                kv_cum = torch.cumsum(kv_, dim=1)
                # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
                qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
                # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
                k_cum = torch.cumsum(k_, dim=1)
                # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
                denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
                # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
                attn_output = qkv / denom.unsqueeze(-1)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)

            else:
                # (N * h, L, 2 * d, d) -> (N * h, 2 * d, d)
                kv_cum = torch.sum(kv_, dim=1)
                # (N * h, 1, 2 * d) (N * h, 2 * d, d) -> (N * h, 1, d)
                qkv = torch.einsum("nld,ndm->nlm", q_, kv_cum)
                # (N * h, L, 2 * d) -> (N * h, 2 * d)
                k_cum = torch.sum(k_, dim=1)
                # (N * h, 1, 2 * d) (N * h, 2 * d) -> (N * h, 1)
                denom = torch.clamp_min(torch.einsum("nlm,nm->nl", q_, k_cum), eps) 
                # (N * h, 1, d) (N * h, 1, 1) -> (N * h, 1, d)
                attn_output = qkv / denom.unsqueeze(-1)
                # (N * h, 1, d) -> (1, N * h, d) -> (L, N, E)
                
        else:
            # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
            kv_ = torch.einsum('nld,nlm->ndm', k_, v)
            # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
            z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
            # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
            attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
            # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        if batch_first:
            attn_output = attn_output.reshape(bsz, num_heads, -1, self.head_dim).transpose(1, 2).reshape(bsz, tgt_len, -1)
        else:
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
                    
        # L, N, E
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)

        return attn_output, None

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
    
    
    def left_product(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
        """Input shape: Sequence x Batch x Embedding
        Args:
            query (Tensor): `(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            key (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            value (Tensor): `(S, N, E)` where S is the source sequence length, N is the batch size,
            E is the embedding dimension.
            attn_mask (Optional[Tensor], optional): typically used to implement causal attention, 
            where the mask prevents the attention from looking forward in time (default: None).
        """
        # test for the correctness of the program
        if key == None:
            key = query
        if value == None:
            value = query
        
        num_heads = self.num_heads
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        head_dim = embed_dim // num_heads

        # get q, k, v
        # (L, N, E)
        q = self.q_proj(query)
        # (S, N, E)
        k = self.k_proj(key)
        # (S, N, E)
        v = self.v_proj(value)

        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        # multihead reshape
        # (N * h, L, d)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # (N * h, S, d)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        
        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        # (N * h, L, d) (N * h, d, S) -> (N * h, L, S)
        weights = torch.bmm(q_, k_.transpose(1, 2))
        # mask
        if self.causal:
            weights = weights.masked_fill(attn_mask==float("-inf"), 0)
        # (N * h, L, S) -> (N * h, L, S)
        denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
        # (N * h, L, S) (N * h, L, S) -> (N * h, L, S)
        attn_weights = weights / denom
        # (N * h, L, S) (N * h, S, d) -> (N * h, L, d)
        attn_output = torch.bmm(attn_weights, v)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
        # L, N, E
        if self.has_outproj:
            attn_output = self.out_proj(attn_output)

        return attn_output

    @staticmethod
    def add_attn_specific_args(parent_parser):
        if hasattr(super(CosformerAttention, CosformerAttention), "add_attn_specific_args"):
            parent_parser = super(CosformerAttention, CosformerAttention).add_attn_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Attention")
        add_nested_argument(parser, '--act-fn', default='relu', type=str,
                                help='activation function for kernel')
        return parent_parser

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


# def main():
#     test(tgt_len=10, src_len=20, causal=False)
#     test(tgt_len=10, src_len=10, causal=True)

# if __name__ == "__main__":
#     main()

