# This file is the implementation of LARA introduced in 
# "Linear Complexity Randomized Self-attention Mechanism"
# Supported pattern of the code: Noncausal Self.
from typing import Optional, Tuple, Dict
import warnings

from torch import Tensor

from efficient_attention import MultiheadAttention, register_cls, add_nested_argument
import torch
import torch.nn as nn
import torch.functional as F


class LinearRA(nn.Module):

    def __init__(self,
                 num_heads,
                 embed_dim,
                 num_landmarks=16,
                 proposal_gen='seg-means',
                 use_antithetics=False,
                 use_opt_approx=False,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = embed_dim
        self.dim_heads = embed_dim // num_heads
        self.landmarks = num_landmarks

        self.proposal_gen = proposal_gen
        self.use_antithetics = use_antithetics
        self.use_opt_approx = use_opt_approx

        if self.proposal_gen == 'adaptive-seg-means':
            self.q_bar_proj = nn.Linear(self.dim_heads, self.dim_heads)
            self.k_bar_proj = nn.Linear(self.dim_heads, self.dim_heads)

    def _segment(self, q, k):
        bars = [None, None]
        for i, x in enumerate([q, k]):
            b, n, d = x.shape  # [B * num_heads, N, D]
            if self.proposal_gen in ['seg-means', 'adaptive-seg-means']:
                segs = n // self.landmarks
                if n < self.landmarks:
                    # It is used for cross attention. For self-attention, we return bar=x directly.
                    # num_k = self.landmarks - N
                    # bar = torch.cat(
                    #     [x for _ in range(self.landmarks // N + 1)], dim=-2
                    # )[:, :self.landmarks]
                    bar = x
                elif n % self.landmarks == 0:
                    bar = x.reshape(b, self.landmarks, segs, d).mean(dim=-2)
                else:
                    num_k = (segs + 1) * self.landmarks - n

                    landmarks_f = x[:, :num_k * segs, :].reshape(
                        b, num_k, segs, d).mean(dim=-2)
                    landmarks_l = x[:, num_k * segs:, :].reshape(
                        b, self.landmarks - num_k, segs + 1, d).mean(dim=-2)
                    bar = torch.cat((landmarks_f, landmarks_l), dim=-2)
            else:
                raise NotImplementedError
            bars[i] = bar
        return bars[0], bars[1]

    def sample_weights(self, q, k):
        assert q.shape[0] == k.shape[0] and q.shape[-1] == k.shape[-1]
        q_bar, k_bar = self._segment(q, k)
        if self.proposal_gen == 'adaptive-seg-means':
            q_bar = self.q_bar_proj(q_bar)
            k_bar = self.k_bar_proj(k_bar)
        mu = q_bar + k_bar
        if self.training:
            if self.use_antithetics:
                noise = torch.randn_like(mu)
                mu = mu.repeat(1, 2, 1)
                weights = mu + torch.cat([noise, -noise], dim=-2)
            else:
                weights = mu + torch.randn_like(mu)
        else:
            weights = mu
        return mu, weights

    def compute_lara(self, q, k, v, mu, weights, key_padding_mask=None):
        b, n, d = q.shape
        log_proj_q = prm_projection(q, weights, normalize=False)  # [b, c, lq]
        log_proj_k = prm_projection(k, weights, normalize=False)  # [b, c, lk]
        log_proj_mu = prm_projection(mu, weights, normalize=False, diagonal=True)  # [b, c, c_mu]
        logit_qk_bar = torch.einsum('...cd,...nd->...cn', (d ** -0.5) * mu, q)  # [b,c,l_q]
        if key_padding_mask is not None:
            log_proj_k_ = log_proj_k.masked_fill(
                        key_padding_mask.unsqueeze(-2).repeat(self.num_heads, 1, 1).to(torch.bool),
                        float("-inf"),
                    )
        else:
            log_proj_k_ = log_proj_k
        kv_stats = torch.einsum(
            '...cm,...md->...cd',
            torch.softmax(log_proj_k_, dim=-1),
            v)  # [b, c, d]
        if self.use_opt_approx:
            log_v_norm = torch.log(torch.linalg.norm(v, ord=2, dim=-1) + 1e-10)  # b n
            stratum_weight = logit_qk_bar + log_v_norm.unsqueeze(-2)
            # [b, c, p] [b, c, 1] [b, c, 1]
            log_ratio = stratum_weight + log_proj_q + \
                torch.logsumexp(log_proj_k, dim=-1, keepdim=True) - \
                log_proj_mu.unsqueeze(-1)  # [b,c,p]
        else:
            stratum_weight = logit_qk_bar
            # [b, c, p] [b, c, 1] [b, c, 1]
            log_ratio = stratum_weight + log_proj_q + \
                torch.logsumexp(log_proj_k, dim=-1, keepdim=True) - \
                log_proj_mu.unsqueeze(-1)  # [b,c,f,p]
        iw = torch.softmax(log_ratio, dim=1)  # [b, c, f, p]
        output = torch.einsum('...cp, ...cd->...pd', iw, kv_stats)
        return output

    def forward(self, q, k, v, key_padding_mask=None):
        mu, weights = self.sample_weights(q, k)
        output = self.compute_lara(q, k, v, mu, weights, key_padding_mask)
        return output, None


def prm_projection(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    normalize: bool = True,
    diagonal: bool = False,
):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    is_query: predicate indicating whether input data corresponds to queries or
        keys
    Returns:
    Random features for fast softmax attention.
    """
    # data : [n, b, h, lk, d]
    # proj : [n, b, h, lc, d]
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    # NOTE: scaler with 0.5 could considerably stable training.
    # now test norm also uses scaled data: that is, multiply by data.shape[-1] ** -1.
    data_normalizer = (data.shape[-1] ** -0.5)
    if diagonal:
        data_dash = torch.einsum('...nd,...nd->...n',
                                 projection_matrix,
                                 (data_normalizer * data),
                                 )  # [n, b, h, lq, lk]
        norm = data_normalizer * torch.sum(data ** 2, dim=-1) / 2.0  # [n, b, h, 1, lk]
    else:
        data_dash = torch.einsum('...nd,...md->...nm',
                                 projection_matrix,
                                 (data_normalizer * data),
                                 )  # [n, b, h, lq, lk]
        norm = data_normalizer * torch.sum(data ** 2, dim=-1).unsqueeze(-2) / 2.0  # [n, b, h, 1, lk]
    if normalize:
        proj_data = F.softmax(data_dash - norm, dim=-1)  # [n, b, h, l_c, l_k]
    else:
        proj_data = data_dash - norm
    return proj_data


@register_cls
class Lara(MultiheadAttention):
    """
    Usage:
    
    from efficient_attention import Lara
    attn = Lara(embed_dim=embed_dim, num_heads=num_heads,num_landmarks=num_landmarks, dropout=dropout)

    result, _ = attn(query, key, value, key_padding_mask=key_padding_mask, batch_first=batch_first, query_padding_mask=query_padding_mask)

    """

    def __init__(self, num_landmarks=16, **kwargs):
        super(Lara, self).__init__(**kwargs)
        assert self.causal == False, f"{self.name.split('.')[0]} cannot do causal attention now"
        assert self.cross == False, f"{self.name.split('.')[0]} cannot do cross attention now"

        self.lara = LinearRA(
            self.num_heads,
            self.embed_dim,
            num_landmarks=num_landmarks
        )

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
        Computes Lara on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q (Tensor): query tensors. :math:`(B, Nt, E)` where B is batch size, Nt is the sequence length of query,
                and E is embedding dimension.
            k (Tensor): key tensors. :math:`(B, Ns, E)` where B is batch size, Nt is the sequence of key,
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
            warnings.warn("`attn_mask` arguments make no sense in `Lara`")
        # assert attn_mask is None, 'causal attention is not supported now!'
        return self.lara(q, k, v, key_padding_mask=key_padding_mask)

    @staticmethod
    def add_attn_specific_args(parent_parser):
        if hasattr(super(Lara, Lara), "add_attn_specific_args"):
            parent_parser = super(Lara, Lara).add_attn_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Attention")

        add_nested_argument(parser, '--num-landmarks', default=15, type=int,
                                help='number of random features')
        
        return parent_parser
