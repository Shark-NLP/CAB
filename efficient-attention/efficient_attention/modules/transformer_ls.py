# This file is the implementation of Transformer LS introduced in 
# "Long-Short Transformer: Efficient Transformers for Language and Vision".
# Supported pattern of the code: Noncausal Self, Causal Self.
# The code comes from https://github.com/NVIDIA/transformer-ls.

import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from efficient_attention import (AbstractAttention, add_nested_argument,
                                 register_cls)
from efficient_attention.modules.multihead_attention import \
    _append_prev_key_padding_mask


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


@register_cls
class AttentionLS(AbstractAttention):
    """
    The long-short term attention for bidirectional language modelling

    Usage:

    from efficient_attention import AttentionLS
    attn = AttentionLS(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, conv_kernel_size=conv_kernel_size, num_landmarks=num_landmarks, window_size=window_size)

    result, _ = attn(query, key_padding_mask, batch_first=batch_first)

    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 head_dim: int = None,
                 num_landmarks: int = 16,
                 conv_kernel_size: int = -1,
                 window_size: int = 15,
                 fp32: bool = True,
                 dropout: float = 0,
                 **kwargs):
        super(AttentionLS, self).__init__(**kwargs)
        assert self.causal == False, f"{self.name.split('.')[0]} cannot do causal attention now"
        assert self.cross == False, f"{self.name.split('.')[0]} cannot do cross-attention now"

        self.cls_from_seq = False

        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.num_landmarks = num_landmarks

        self.dim = embed_dim

        self.drop_attn = torch.nn.Dropout(p=dropout)

        self.window_size = window_size

        self.W_q = nn.Linear(self.dim, self.num_heads * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_heads * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_heads * self.head_dim)
        self.W_o = nn.Linear(self.dim, self.num_heads * self.head_dim)

        self.fp32 = fp32

        self.dual_ln_s = nn.LayerNorm(self.num_heads * self.head_dim)
        self.dual_ln_l = nn.LayerNorm(self.num_heads * self.head_dim)

        self.dconv_fc = nn.Linear(self.dim, self.num_heads * self.num_landmarks)

        self.use_conv = conv_kernel_size > 0
        if self.use_conv:
            self.conv = nn.Conv1d(
                in_channels=self.num_heads, out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1), padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads)
            nn.init.zeros_(self.conv.weight)

    def get_tiles(self, x, transpose=False):
        # x: bsz x n_heads x seqlen x d_head
        bsz, n_heads, seqlen, d_h = x.shape
        out_shape = (bsz, n_heads, seqlen // self.window_size - 1, 2 * self.window_size, d_h)
        in_strides = x.stride()
        out_strides = (in_strides[0], in_strides[1], in_strides[2] * self.window_size, in_strides[2], 1)

        x_main = x.as_strided(size=out_shape, stride=out_strides)
        x_last = x[:, :, None, -2 * self.window_size:, :]
        x = torch.cat([x_main, x_last], dim=2)
        if transpose:
            return x.transpose(-1, -2)
        else:
            #  bsz x n_heads x seqlen//wlen x 2*wlen x d_h
            return x

    def get_tiled_mask(self, mask):
        bsz, seqlen = mask.shape
        out_shape = (bsz, seqlen // self.window_size - 1, 2 * self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1] * self.window_size, in_stride[1])
        mask_main = mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, :]
        mask_last = mask[:, None, None, -2 * self.window_size:]

        return torch.cat([mask_main, mask_last], dim=2)[:, :, :, None, :]

    def sliding_chunks_matmul_qk(self, Q, K, padding_mask):
        # Q, K: bsz x num_heads x seqlen x d_head
        # padding_mask: bsz x seqlen
        bsz, num_heads, seqlen, d_h = Q.shape
        mask_tiles = self.get_tiled_mask(padding_mask)
        K_tiles = self.get_tiles(K, transpose=True)
        Q_tiles = Q.view(bsz, num_heads, seqlen // self.window_size, self.window_size, d_h)
        # bsz x num_heads x seqlen//winsize x winsize x 2winsize
        qk_scores = Q_tiles.matmul(K_tiles)
        qk_scores.masked_fill_(mask_tiles, float('-inf'))
        return qk_scores.view(bsz, num_heads, seqlen, 2 * self.window_size)

    def get_tiles_v2(self, x, transpose=False):
        if self.window_size <= 0:
            return x

        bsz, n_heads, seqlen, d_h = x.shape
        n_groups = seqlen // self.window_size
        ext_len = max(self.window_size // 2, 1)
        x = F.pad(x, (0, 0, ext_len, ext_len), value=0)
        strides = x.stride()
        if transpose:
            out_shape = (bsz, n_heads, n_groups, d_h, 2 * ext_len + self.window_size)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[3], strides[2])
        else:
            out_shape = (bsz, n_heads, n_groups, 2 * ext_len + self.window_size, d_h)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)

    def get_tiled_mask_v2(self, mask):
        # only mask along the key dimension
        bsz, seqlen = mask.shape
        ext_len = max(self.window_size // 2, 1)
        mask = F.pad(mask, (ext_len, ext_len), value=True)
        out_shape = (bsz, seqlen // self.window_size, 2 * ext_len + self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1] * self.window_size, in_stride[1])
        return mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, None, :]

    def sliding_chunks_matmul_qk_v2(self, Q, K, padding_mask):
        bsz, num_heads, seqlen, d_h = Q.shape
        if self.window_size > 0:
            # Q, K: bsz x num_heads x seqlen x d_head
            # padding_mask: bsz x seqlen

            mask_tiles = self.get_tiled_mask_v2(padding_mask)
            K_tiles = self.get_tiles_v2(K, transpose=True)
            Q_tiles = Q.view(bsz, num_heads, seqlen // self.window_size, self.window_size, d_h)
            # bsz x num_heads x seqlen//winsize x winsize x 2winsize
            qk_scores = Q_tiles.matmul(K_tiles)
            qk_scores = qk_scores.masked_fill(mask_tiles, float('-inf'))
            return qk_scores.view(bsz, num_heads, seqlen, -1)
        else:
            qk_scores = torch.sum(Q * K, dim=-1, keepdim=True)
            return qk_scores

    def forward(self, query: torch.Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_padding_mask: Tensor = None,
                key_padding_mask: torch.Tensor = None,
                need_weights=None,
                need_head_weights: bool = False,
                attn_mask: Tensor = None,
                batch_first=False,
                static_kv: bool = False,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                **kwargs):
        warnings.warn("`attn_mask` arguments make no sense in `AttentionLS`")
        assert not (self.num_landmarks <= 0 and self.window_size <= 0)

        if not batch_first:
            query = query.transpose(0, 1)
        bsz, seqlen, d_model = query.shape
        if key_padding_mask is None:
            padding_mask = torch.zeros((bsz, seqlen)).to(query.device)
        else:
            padding_mask = key_padding_mask
        padded = False
        if seqlen % self.window_size:
            orig_len = seqlen
            query = F.pad(query, [0, 0, 0, (self.window_size - seqlen % self.window_size)])
            padding_mask = F.pad(padding_mask, [0, (self.window_size - seqlen % self.window_size)], value=1)
            padded = True
        bsz, seqlen, d_model = query.shape
        padding_mask = padding_mask.bool()

        # bsz x n_head x length x head_dim
        Q = self.split_heads(self.W_q(query)).mul(1. / math.sqrt(self.head_dim))

        K = self.split_heads(self.dual_ln_l(self.W_k(query)))
        V = self.split_heads(self.dual_ln_l(self.W_v(query)))

        if self.fp32:
            Q, K, V = Q.float(), K.float(), V.float()

        # bsz x length x num_head*num_lms

        K_compress = V_compress = None
        if self.num_landmarks > 0:
            head_scores = self.dconv_fc(query).masked_fill(padding_mask[:, :, None], float('-inf'))
            head_scores = F.softmax(head_scores, dim=1, dtype=torch.float32)  # .to(X)
            if not self.fp32:
                head_scores = head_scores.to(query)
            # bsz x num_head x num_lms x length
            head_scores = head_scores.view(bsz, seqlen, self.num_heads, self.num_landmarks).permute(0, 2, 3, 1)
            K_compress = head_scores.matmul(K)
            V_compress = head_scores.matmul(V)

        if self.dual_ln_s is not None and K_compress is not None:
            K_compress = self.dual_ln_s(K_compress.transpose(1, 2).contiguous().view(bsz, -1, d_model))
            K_compress = self.split_heads(K_compress)
            V_compress = self.dual_ln_s(V_compress.transpose(1, 2).contiguous().view(bsz, -1, d_model))
            V_compress = self.split_heads(V_compress)

        if self.num_landmarks > 0:
            # bsz x num_head x length x num_lms
            attn_compress = Q.matmul(K_compress.transpose(-1, -2))
        else:
            attn_compress = None

        if self.window_size > 0 or self.num_landmarks == 0:
            # First, compute the compressed part, or the attentions on the landmarks
            # First use window attention to attend to the diagonals
            # V: bsize, self.seq_len, self.num_head, self.head_dim
            # win_attn_weights = self.sliding_chunks_matmul_qk(Q, K, padding_mask)
            win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K, padding_mask)
        else:
            win_attn_weights = None

        if attn_compress is None:
            all_attn_ = win_attn_weights
        elif win_attn_weights is None:
            all_attn_ = attn_compress
        else:
            all_attn_ = torch.cat([attn_compress, win_attn_weights], dim=-1)

        all_attn = all_attn_.float().softmax(dim=-1).to(win_attn_weights)
        # If one of the rows are all -inf, then it will be NaN!
        all_attn = all_attn.masked_fill(padding_mask[:, None, :, None], 0)
        if not self.fp32:
            all_attn = all_attn.to(query)
        all_attn = self.drop_attn(all_attn)

        C = 0
        if attn_compress is not None:
            C += all_attn[:, :, :, :K_compress.shape[2]].matmul(V_compress)

        if win_attn_weights is not None:
            win_attn_probs = all_attn[:, :, :, -win_attn_weights.shape[-1]:]
            if self.window_size > 0:
                win_attn_probs = win_attn_probs.view(bsz, self.num_heads, seqlen // self.window_size, self.window_size,
                                                     -1)
                V_tiles = self.get_tiles_v2(V, transpose=False)
                C += win_attn_probs.matmul(V_tiles).view(bsz, self.num_heads, seqlen, self.head_dim)
            else:
                C += win_attn_probs * V

        if self.use_conv:
            V = V.masked_fill(padding_mask[:, None, :, None], 0)
            C = C + self.conv(V)

        if self.fp32:
            # Finally convert it back, same as Nystromformer
            C = C.to(query)
        out = self.W_o(self.combine_heads(C))
        if padded:
            out = out[:, :orig_len, :]
        if not batch_first:
            out = out.transpose(0, 1)
        return out, None

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, window_size={self.window_size}'

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_heads * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_heads, self.head_dim)
        X = X.transpose(1, 2)
        return X

    @staticmethod
    def add_attn_specific_args(parent_parser):
        if hasattr(super(AttentionLS, AttentionLS), "add_attn_specific_args"):
            parent_parser = super(AttentionLS, AttentionLS).add_attn_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Attention")
        add_nested_argument(parser, '--num-landmarks', default=16, type=int)
        add_nested_argument(parser, '--conv-kernel-size', default=-1, type=int)
        add_nested_argument(parser, '--window-size', default=False, type=int)
        add_nested_argument(parser, '--fp32', default=False, action='store_true')
        return parent_parser


@register_cls
class CausalLS(AbstractAttention):
    r"""
    Usage:

    from efficient_attention import CausalLS

    attn = CausalLS(embed_dim=embed_dim, num_heads=num_heads, chunk_size=chunk_size, window_len=window_len, dropout=dropout, mem_len=4096)
    result, _ = attn(src_tokens, incremental_state=incremental_state, batch_first=batch_first)

    NOTE `chunk_size` is designed for parallelization of causal attention, so make it divisible by mem_len(1024 by default)
    incremental_state is a special design in `fairseq` for incremental decoding. We will add this function later.

    """

    def __init__(self, embed_dim, num_heads, chunk_size=32, window_size=256, dropout=0.0, chunk_rank=1,
                 mem_len=1024,
                 grad_chk=False, use_bias=False, dp_attn=0, **kwargs) -> None:
        super(CausalLS, self).__init__(**kwargs)
        assert self.cross == False, f"{self.name.split('.')[0]} cannot do cross-attention now"
        self.causal = True
        self.dropout = nn.Dropout(dropout)
        self.dp_attn = nn.Dropout(dp_attn)

        assert embed_dim % num_heads == 0
        assert chunk_size > 0
        # because this causal attention calculates the causal attention in chunk grain, the sequence length must be divisible by the chunk
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_len = window_size

        self.chunk_rank = chunk_rank
        self.chunk_size = chunk_size

        self.d_h = embed_dim // num_heads
        self.d_model = embed_dim
        self.mem_len = mem_len
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.dconv_1 = nn.Linear(embed_dim, num_heads * chunk_rank)

        self.r_net = nn.Linear(embed_dim, embed_dim, bias=False)
        self.r_net_chunk = nn.Linear(embed_dim, embed_dim)
        self.d_head = embed_dim // self.num_heads
        # Positional bias as in Transformer-XL.
        self.r_r_bias = nn.Parameter(torch.FloatTensor(1, self.num_heads, 1, self.d_head))
        self.r_w_bias = nn.Parameter(torch.FloatTensor(1, self.num_heads, 1, 1, self.d_head))

        self.grad_chk = grad_chk

        self.proj_query = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        nn.init.xavier_normal_(self.proj_query.weight)
        self.proj_out = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        nn.init.xavier_normal_(self.proj_out.weight)
        self.proj_val = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        nn.init.xavier_normal_(self.proj_val.weight)
        self.proj_key = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        nn.init.xavier_normal_(self.proj_key.weight)

        self.dual_ln_dproj = nn.LayerNorm(embed_dim)
        self.dual_ln_win = nn.LayerNorm(embed_dim)
        self.mems = None
        nn.init.zeros_(self.r_r_bias)
        nn.init.zeros_(self.r_w_bias)
        if use_bias:
            nn.init.zeros_(self.proj_query.bias)
            nn.init.zeros_(self.proj_out.bias)
            nn.init.zeros_(self.proj_val.bias)
            nn.init.zeros_(self.proj_key.bias)

    def head_reshape(self, x):
        K = self.num_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        return x

    def compute_scores(self, h_vecs):
        # h_vecs: B x L x H
        bsz = h_vecs.shape[0]
        n_chunks = h_vecs.shape[1] // self.chunk_size
        h_scores = self.dconv_1(h_vecs).view(bsz, n_chunks, self.chunk_size, self.num_heads, self.chunk_rank)
        # bsz x num_heads x n_chunks x chunk_rank x chunk_size
        h_scores = h_scores.permute(0, 3, 1, 4, 2)
        h_scores = F.softmax(h_scores.float(), dim=-1).type_as(h_scores)
        return h_scores

    def compress_chunks(self, h_vecs, h_scores):
        # Reshape hvecs to be compatible with the weights
        # h_vecs: B x L x H
        bsz = h_vecs.shape[0]
        n_chunks = h_vecs.shape[1] // self.chunk_size
        # bsz x n_heads x n_chunks x chunk_size x d_h
        h_vecs = h_vecs.view(-1, n_chunks, self.chunk_size, self.num_heads, self.d_h).permute(0, 3, 1, 2, 4)
        # bsz x n_heads x n_chunks x chunk_rank x d_h
        h_vecs = h_scores.matmul(h_vecs).view(bsz, self.num_heads, n_chunks * self.chunk_rank, self.d_h)
        return h_vecs

    def get_tiles(self, x: Tensor, n_queries, transpose=False):
        # input: bsz x win_bp_len x d
        bsz, win_bp_len, d = x.shape
        in_strides = x.stride()
        out_strides = (in_strides[0], self.window_len * in_strides[1], in_strides[1], d // self.num_heads, 1)
        out_shape = (bsz, n_queries // self.window_len, 2 * self.window_len, self.num_heads, d // self.num_heads)
        x = x.as_strided(size=out_shape, stride=out_strides)
        if transpose:
            # shape: bsz x n_heads x n_queries//wlen x d//n_heads x 2*wlen
            return x.permute(0, 3, 1, 4, 2)
        else:
            # shape: bsz x n_heads x n_queries//wlen x 2*wlen x d//n_heads
            return x.permute(0, 3, 1, 2, 4)

    def put_tiles(self, x):
        # input: bsz x n_heads x bp_len x self.window_len
        bsz, n_heads, bp_len, window_len = x.shape
        if bp_len > window_len:
            x = x.view(bsz, n_heads, bp_len // window_len, window_len, window_len)
            out_size = (bsz, n_heads, bp_len // window_len, window_len, 2 * window_len)
            x = F.pad(x, (1, window_len))
        else:
            x = x.view(bsz, n_heads, 1, bp_len, window_len)
            out_size = (bsz, n_heads, 1, bp_len, window_len + bp_len)
            x = F.pad(x, (1, bp_len))

        stride = x.stride()
        out_stride = (stride[0], stride[1], stride[2], stride[3] - 1, stride[4])
        return x.as_strided(size=out_size, stride=out_stride)

    def compute_pv(self, attn, val):
        # attn: bsz x n_head x seqlen//wlen x wlen x 2*wlen
        # val:  bsz x n_head x seqlen//wlen x 2*wlen x d_h
        bsz, n_head, chunks, wlen, _ = attn.shape
        out = attn.matmul(val)
        return out.view(bsz, n_head, int(chunks * wlen), -1)

    def get_diagonals(self, attn):
        # attn:  bsz x n_heads x bp_len//self.window_len x self.window_len x 2*self.window_len
        # takes the upper diagonal with length self.window_len from attn, ignoring the diagonal
        bsz, n_heads, n_tiles, n_query, _ = attn.shape
        out_size = (bsz, n_heads, n_tiles, n_query, self.window_len)
        in_stride = attn.stride()
        out_stride = (in_stride[0], in_stride[1], in_stride[2], in_stride[3] + 1, 1)
        return attn.as_strided(size=out_size, stride=out_stride, storage_offset=1).contiguous().view(
            bsz, n_heads, -1, self.window_len)

    def _rel_shift_chunked(self, x, chunk_size, chunk_rank):
        # x: bsz x n_head x n_query x (n_chunks * chunk_rank)
        # out: same size but shifted to the left, relative position encoding
        bsz, n_head, n_query, n_c_vecs = x.shape
        n_q_chunks = n_query // chunk_size
        x = x.view(bsz, n_head, n_q_chunks, chunk_size, n_c_vecs).transpose(2, 3).contiguous()
        x = F.pad(x, [0, chunk_rank])
        p_stride = x.stride()
        out_shape = list(x.shape)
        out_shape[-1] -= chunk_rank
        out_strides = (p_stride[0], p_stride[1], p_stride[2], p_stride[3] - chunk_rank, p_stride[4])

        x = x.as_strided(size=out_shape, stride=out_strides, storage_offset=n_q_chunks * chunk_rank)
        return x.transpose(2, 3).contiguous().view(bsz, n_head, n_query, n_c_vecs)

    def attn(self, query, key_window, val_window, key_compressed, value_compressed,
             pos_embed_chunks, pos_embed_window, chunk_attn_mask=None):
        # query size = bsz x n_heads x M x H
        # key, value sizes = bsz x (seq_len + cache_len) x (n_heads * H)
        # key_compressed: bsz x n_heads x (M+L)//chunk_size*chunk_rank x H
        bsz, n_heads, seq_len, d_h = query.shape
        assert (self.window_len > 0 or self.chunk_size > 1)

        query = query / math.sqrt(self.d_model // self.num_heads)

        # get the keys, values for the local window attention
        if seq_len > self.window_len:
            query_tile = query.view(bsz, n_heads, seq_len // self.window_len, self.window_len, d_h)
            key_window = self.get_tiles(key_window, seq_len, transpose=True)
            val_window = self.get_tiles(val_window, seq_len,
                                        transpose=False)  # bsz x n_heads x n_queries//wlen x 2*wlen x d//n_heads
        else:
            query_tile = query.view(bsz, n_heads, 1, seq_len, d_h)
            key_window = key_window.view(bsz, -1, self.num_heads, d_h).permute(0, 2, 3, 1)[:, :, None, :, :]
            val_window = val_window.view(bsz, -1, self.num_heads, d_h).permute(0, 2, 1, 3)[:, :, None, :, :]
        # bsz x n_heads x bp_len//self.window_len x self.window_len x 2*self.window_len
        attn_window = (query_tile + self.r_w_bias).matmul(key_window)
        # print(attn_window, attn_window.shape)
        attn_window = self.get_diagonals(attn_window)

        pos_trans = self.r_net(pos_embed_window).view(1, self.window_len, self.num_heads, self.d_head).permute(0, 2, 3,
                                                                                                               1)
        attn_window_pos = (query + self.r_r_bias).matmul(pos_trans)
        attn_window = attn_window + attn_window_pos
        # Compute the long-range attention.
        n_chunks = key_compressed.shape[2]
        # compute attention from context
        # bsz x n_heads x seq_len x (n_chunks*chunk_rank)
        attn_cont = torch.matmul(query, key_compressed.transpose(-1, -2))
        pos_chunks = self.r_net_chunk(pos_embed_chunks).view(1, n_chunks, self.num_heads, self.d_head).permute(0, 2, 3,
                                                                                                               1)

        attn_pos = torch.matmul(query, pos_chunks)  # B x H x M x L_pos
        attn_pos = self._rel_shift_chunked(attn_pos, self.chunk_size, self.chunk_rank)

        attn_compress = attn_cont + attn_pos
        if chunk_attn_mask is not None:
            attn_compress = attn_compress.view(
                bsz, n_heads, seq_len // self.chunk_size, self.chunk_size, -1)
            attn_compress = attn_compress.masked_fill(chunk_attn_mask, float('-inf'))
            attn_compress = attn_compress.view(bsz, n_heads, seq_len, -1)

        # Get the softmax score of both short-term and long-range attentions.
        # print(attn_window)
        full_attn = torch.cat([attn_compress, attn_window], dim=3)
        # print(full_attn)
        full_attn = F.softmax(full_attn.float(), dim=-1).type_as(full_attn)

        full_attn = self.dp_attn(full_attn)

        attn_compress = full_attn[:, :, :, :attn_compress.shape[3]]
        # print(attn_compress)
        attn_window = full_attn[:, :, :, attn_compress.shape[3]:]

        attn_window = self.put_tiles(attn_window)
        out = torch.matmul(attn_compress, value_compressed) \
              + self.compute_pv(attn_window, val_window)

        return out

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_padding_mask=None,
                key_padding_mask=None,
                need_weights: bool = True,
                need_head_weights: bool = False,
                attn_mask=None,
                static_kv: bool = False,
                incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                batch_first=False,
                **kwargs):
        if attn_mask is not None:
            warnings.warn("`attn_mask` arguments make no sense in `CausalLS`")
        if self.grad_chk:
            out = cp.checkpoint(self.forward_, *[
                query, key, value, incremental_state, batch_first, static_kv, key_padding_mask
            ])
        else:
            out = self.forward_(query, key, value, incremental_state, batch_first, static_kv, key_padding_mask)
        return out, None

    def forward_(self, query: Tensor,
                 key: Tensor = None,
                 value: Tensor = None,
                 incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                 batch_first=False,
                 static_kv=False,
                 key_padding_mask: Optional[Tensor] = None):
        # make sure that incremental state is not None
        if key is None:
            key = query
        if value is None:
            value = query
        if not batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        bsz = query.shape[0]

        # project to get key and value

        if incremental_state is not None:
            saved_state: Optional[Dict[str, Optional[Tensor]]] = {}
            key, value = self._get_saved_states(incremental_state, saved_state, static_kv, key, value)
        else:
            saved_state: Optional[Dict[str, Optional[Tensor]]] = None

        if saved_state is not None:
            save_key = key.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).reshape(bsz * self.num_heads,
                                                                                                   -1, self.head_dim)
            save_val = value.reshape(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).reshape(
                bsz * self.num_heads, -1, self.head_dim)
            key, value, key_padding_mask = self._update_saved_states(save_key, save_val, key_padding_mask, saved_state,
                                                                     bsz, static_kv)
            key = key.reshape(bsz, self.num_heads, -1, self.head_dim).transpose(1, 2).reshape(bsz, -1, self.d_model)
            value = value.reshape(bsz, self.num_heads, -1, self.head_dim).transpose(1, 2).reshape(bsz, -1, self.d_model)
            incremental_state[self.name] = saved_state
        # x = bsz x seq_len x H
        # h_cache = bsz x cache_len x H
        padded = False

        # orig_seqlen = query.shape[1]
        if self.chunk_size > 0 and (query.shape[1] % self.chunk_size):
            orig_seqlen = query.shape[1]
            pad_multip = abs(self.chunk_size * self.window_len) // math.gcd(self.chunk_size, self.window_len)
            n_pad = pad_multip - query.shape[1] % pad_multip
            query = F.pad(query, (0, 0, 0, n_pad))
            padded = True

        val_window_bp = self.proj_val(query)
        key_window_bp = self.proj_key(query)

        h_cache = torch.zeros((bsz, self.mem_len, self.d_model)).to(query)

        mlen = self.mem_len
        klen = query.shape[1] + mlen
        n_chunk_vecs = klen // self.chunk_size * self.chunk_rank
        n_chunks = klen // self.chunk_size
        n_mem_chunks = mlen // self.chunk_size
        chunk_attn_mask = torch.triu(query.new_ones((query.shape[1] // self.chunk_size, n_chunks), dtype=torch.bool),
                                     diagonal=n_mem_chunks)[
                          None, None, :, None, :, None]
        chunk_attn_mask = chunk_attn_mask.expand(-1, -1, -1, -1, -1, self.chunk_rank).contiguous().view(1, 1, -1, 1,
                                                                                                        n_chunks * self.chunk_rank)
        pos_chunk_ids = torch.arange(n_chunk_vecs - 1, -1, -1.0, device=query.device, dtype=query.dtype)
        pos_seq = torch.arange(self.window_len - 1, -1, -1.0, device=query.device, dtype=query.dtype)
        key_pe = self.pos_emb(pos_chunk_ids)
        pos_embed_window = self.pos_emb(pos_seq)

        seqlen = query.shape[1]
        q = self.proj_query(query)
        q = self.head_reshape(q)

        # sequence length and cache length should be divisible by the chunk size
        assert seqlen % self.chunk_size == 0 and h_cache.shape[1] % self.chunk_size == 0

        # better using multipliers of 8
        ##################### window attention #######################
        h_cache_win = h_cache[:, -self.window_len:]
        # h_cache_win = torch.zeros((bsz, self.window_len, self.d_model)).to(q)
        key_cache_win = self.proj_key(h_cache_win)
        val_cache_win = self.proj_val(h_cache_win)

        key_window = torch.cat([key_cache_win, key_window_bp], dim=1)
        val_window = torch.cat([val_cache_win, val_window_bp], dim=1)
        # DualLN (window)

        key_window = self.dual_ln_win(key_window)
        val_window = self.dual_ln_win(val_window)

        ##################### window attention #######################

        ############### dynamic projection attention #################
        # dynamic projection
        cache_scores = self.compute_scores(h_cache)
        h_cache_compressed = self.compress_chunks(h_cache, cache_scores)

        # The projection for the cache can be compressed using dynamic projection
        h_cache_merge = h_cache_compressed.view(
            bsz, self.num_heads, -1, self.d_h).transpose(1, 2).contiguous().view(
            bsz, -1, self.d_model)
        # b x n_chunk x D

        # Apply projections to the compressed sequence.
        val_cache = self.proj_val(h_cache_merge)
        key_cache = self.proj_key(h_cache_merge)
        # DualLN (dproj)
        key_cache = self.dual_ln_dproj(key_cache)
        val_cache = self.dual_ln_dproj(val_cache)
        # b x h x n_chunk x d_h
        val_cache = self.head_reshape(val_cache)
        key_cache = self.head_reshape(key_cache)

        bp_scores = self.compute_scores(query)
        # Compress the projeced keys and values.
        val_bp_compressed = self.compress_chunks(val_window_bp, bp_scores)
        key_bp_compressed = self.compress_chunks(key_window_bp, bp_scores)

        # DualLN (dproj)
        val_bp_compressed = self.dual_ln_dproj(
            val_bp_compressed.transpose(1, 2).contiguous().view(bsz, -1, self.d_model))
        key_bp_compressed = self.dual_ln_dproj(
            key_bp_compressed.transpose(1, 2).contiguous().view(bsz, -1, self.d_model))
        val_bp_compressed = self.head_reshape(val_bp_compressed)
        key_bp_compressed = self.head_reshape(key_bp_compressed)
        # b x h x n_chunk x d_h

        val_compressed = torch.cat([val_cache, val_bp_compressed], dim=2)
        key_compressed = torch.cat([key_cache, key_bp_compressed], dim=2)
        # print(q.shape, key_window.shape, val_window.shape, key_compressed.shape, val_compressed.shape, key_bp_compressed.shape)
        out = self.attn(q, key_window, val_window, key_compressed, val_compressed, key_pe, pos_embed_window,
                        chunk_attn_mask)  # B_K x M x D

        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(bsz, seqlen, -1)  # B x M x K_D
        if padded:
            out = out[:, :orig_seqlen]
        if not batch_first:
            out = out.transpose(0, 1)
        out = self.proj_out(out)
        out = self.dropout(out)
        return out

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

        # saved_state['prev_key_before_proj'] = key
        saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
        saved_state["prev_key_padding_mask"] = key_padding_mask

        return k, v, key_padding_mask

    @staticmethod
    def add_attn_specific_args(parent_parser):
        if hasattr(super(CausalLS, CausalLS), "add_attn_specific_args"):
            parent_parser = super(CausalLS, CausalLS).add_attn_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("Attention")
        add_nested_argument(parser, '--chunk-size', default=32, type=int)
        add_nested_argument(parser, '--mem-len', default=-1, type=int)
        add_nested_argument(parser, '--window-size', default=256, type=int)
        add_nested_argument(parser, '--chunk-rank', default=1, type=int)
        return parent_parser


if __name__ == "__main__":
    embed_dim = 32
    bsz = 1
    num_heads = 1
    seq_len = 4096
    x = torch.randn((bsz, seq_len, embed_dim))
    # attn = AttentionLS(embed_dim=embed_dim, num_heads=num_heads)
    # y, _ = attn(x, batch_first=True)
    # print(y, y.shape)
    mem_size = 4096
    layers = 1
    attns = [CausalLS(embed_dim=embed_dim, num_heads=num_heads, window_len=15, chunk_size=32, dropout=0.2) for l in
             range(layers)]
    incremental_states = [None for l in range(layers)]
    predicted = F.pad(x, (0, 0, 0, 1))
    for i in range(seq_len):
        print("Processing Step {}".format(i + 1))
        x_ = predicted[:, i: i + 1, :]
        for l in range(layers):
            x_, _ = attns[l](x_, x_, x_, incremental_state=incremental_states[l], batch_first=True)

        predicted[:, i + 1, :] = x_[:, 0, :]

