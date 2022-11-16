# This file is the implementation of Performer introduced in 
# "RETHINKING ATTENTION WITH PERFORMERS."
# Supported pattern of the code: Noncausal Self, Noncausal Cross, Causal Cross.
# The code comes from https://github.com/google-research/google-research/tree/master/performer.
from typing import Dict, Optional, Tuple
import warnings

import torch
from torch import Tensor

from efficient_attention import MultiheadAttention, register_cls, add_nested_argument


def orthogonal_matrix_chunk(cols, device = None, dtype=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t().to(dtype)


def create_proj_matrix(num_heads, proj_dim, input_dim, ortho=False, seed=0, device=None, dtype=None):
    if ortho:
        return torch.stack(
            [
                gaussian_orthogonal_random_matrix(proj_dim, input_dim, seed=seed + h * 1000, device=device, dtype=dtype)
                for h in range(num_heads)
            ], dim=0)
    else:
        return torch.randn(num_heads, proj_dim, input_dim, device=device, dtype=dtype)


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, seed=0, device=None, dtype=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []
    cur_seed = seed

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device, dtype=dtype)
        block_list.append(q)
        cur_seed = cur_seed + 1

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device, dtype=dtype)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    multiplier = torch.randn((nb_rows, nb_columns), device=device, dtype=dtype).norm(dim=1)

    return torch.diag(multiplier) @ final_matrix


def softmax_kernel(data, is_query, projection_matrix, eps=1e-4):
    data_normalizer = (data.shape[-1] ** -0.25)

    ratio = (projection_matrix.shape[1] ** -0.5)
    data_proj = torch.einsum('bhpd,hmd->bhpm', (data_normalizer * data), projection_matrix.type_as(data))

    data_norm = (torch.sum(data ** 2, dim=-1, keepdim=True) / 2.0) * (data_normalizer ** 2)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_proj - data_norm - torch.amax(data_proj, dim=-1, keepdim=True)) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_proj - data_norm - torch.amax(data_proj, dim=(-1, -2), keepdim=True)) + eps)  # note..

    return data_dash.type_as(data)


@register_cls
class Performer(MultiheadAttention):
    r"""
    Usage:
    
    from efficient_attention import Performer
    attn = Performer(embed_dim=embed_dim, num_heads=num_heads,approx_attn_dim=approx_attn_dim, dropout=dropout)

    result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask)
    
    """
    
    def __init__(self,
                 approx_attn_dim=16,
                 **kwargs):
        super(Performer, self).__init__(**kwargs)

        self.approx_attn_dim = approx_attn_dim

        if self.causal:
            warnings.warn("Performer is incompetent at self causal attention for causal leakage. Please unset this argument")
        self.causal = False
        self.eval_proj = create_proj_matrix(
            self.num_heads, self.approx_attn_dim, self.head_dim, ortho=True)

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
        r"""
        Computes Performer attention on query, key and value tensors, using
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
            warnings.warn("`attn_mask` arguments make no sense in `Performer`")
        tgt_len, src_len = q.shape[1], k.shape[1]

        if self.training:
            projection_matrix = create_proj_matrix(
                self.num_heads, self.approx_attn_dim, self.head_dim, ortho=False, device=q.device, dtype=q.dtype)
        else:
            projection_matrix = self.eval_proj

        q_prime = softmax_kernel(
            q.reshape(bsz, self.num_heads, tgt_len, self.head_dim),
            is_query=True,
            projection_matrix=projection_matrix
        ).reshape(bsz * self.num_heads, tgt_len, -1)
        k_prime = softmax_kernel(
            k.reshape(bsz, self.num_heads, src_len, self.head_dim),
            is_query=False,
            projection_matrix=projection_matrix
        ).reshape(bsz * self.num_heads, src_len, -1)
        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if key_padding_mask is not None:
            k_prime = k_prime * (1 - key_padding_mask.unsqueeze(-1).repeat(self.num_heads, 1, 1).type_as(k_prime))
        if not self.causal:
            kv = torch.einsum('bnm,bnd->bmd', k_prime, v)
            qkv = torch.einsum('bnm,bmd->bnd', q_prime, kv)
            normaliser = torch.einsum('bnm,bm->bn', q_prime, k_prime.sum(dim=-2))
            output = qkv / normaliser.unsqueeze(-1).clamp(min=1e-1)
        else:
            # NOTE: has causal leakage problem
            output = causal_linear_attention_noncuda(q_prime, k_prime, v)
            if incremental_state is not None:
                output = output[:, -1:]
        return output, None
    
    @staticmethod
    def add_attn_specific_args(parent_parser):
        if hasattr(super(Performer, Performer), "add_attn_specific_args"):
            parent_parser = super(Performer, Performer).add_attn_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Attention")

        add_nested_argument(parser, '--approx-attn-dim', default=16, type=int,
                                help='number of random features')
        return parent_parser


# reference: https://github.com/pkuzengqi/Skyformer
def causal_linear_attention_noncuda(q, k, v):
    kv = torch.einsum('bnm,bnd->bnmd', k, v)
    # kv = kv.cumsum(1)
    qkv = torch.einsum('bnm,bnmd->bnd', q, kv)
    normaliser = torch.einsum('bnm,bnm->bn', q, k.cumsum(dim=-2))
    out = qkv / normaliser.unsqueeze(-1).clamp(min=1e-1)
    # k_cumsum = k.cumsum(dim=-2)
    # D_inv = 1. / torch.einsum('...nm,...nm->...n', q, k_cumsum.type_as(q))
    # context = torch.einsum('...nm,...nd->...nmd', k, v)
    # context = context.cumsum(dim=-3)
    # out = torch.einsum('...nmd,...nm,...n->...nd', context, q, D_inv)
    return out
