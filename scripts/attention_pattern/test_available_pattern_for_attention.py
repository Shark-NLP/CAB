import torch

import numpy
import random

from efficient_attention import LocalAttention, ABC, S4D


def seed():
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)


def test_self_causal(attn):
    attn.eval()

    input_ids = torch.randn((batch_size, tgt_len, dim))

    try:
        out = attn(input_ids, batch_first=True)[0]

        for i in range(tgt_len//2, tgt_len):
            z = out[:, i, :]
            x = attn(input_ids[:, :i + 1, :], batch_first=True)[0][:, i, :]
            if (x - z).abs().sum() > 1e-4:
                return False
    except:
        return False
    return True


def test_self_causal_incremental(attn):
    attn.eval()

    input_ids = torch.randn((batch_size, tgt_len, dim))
    incremental_state = {attn.name: {}}
    try:
        out = attn(input_ids, batch_first=True)[0]

        for i in range(0, tgt_len):
            z = out[:, i, :]
            x_ = input_ids[:, i:i+1, :]
            x = attn(x_, x_, x_, batch_first=True, incremental_state=incremental_state)[0][:, 0]
            # NOTE: we use fast implementation for incremental ABC causal self, 
            # which loses some precision and causes failure here
            # Deep into (x-z).abs().sum(), ABC produces just 5e-4, which is acceptable, though.
            if (x - z).abs().sum() > 1e-4:
                return False
    except:
        return False
    return True


def test_cross_causal(attn):
    attn.eval()

    input_ids = torch.randn((batch_size, tgt_len, dim))
    key = torch.randn((batch_size, src_len, dim))
    value = torch.randn((batch_size, src_len, dim))

    try:
        out = attn(input_ids, key, value, batch_first=True)[0]

        for i in range(tgt_len//2, tgt_len):
            z = out[:, i, :]
            x = attn(input_ids[:, :i + 1, :], key, value, batch_first=True)[0][:, i, :]
            if (x - z).abs().sum() > 1e-4:
                return False
    except:
        return False
    return True


def test_cross_noncausal(attn):
    attn.eval()

    input_ids = torch.randn((batch_size, tgt_len, dim))
    key = torch.randn((batch_size, src_len, dim))
    value = torch.randn((batch_size, src_len, dim))

    try:
        out = attn(input_ids, key, value, batch_first=True)[0]
    except:
        return False
    return True


def test_self_noncausal(attn):
    attn.eval()

    input_ids = torch.randn((batch_size, tgt_len, dim))

    try:
        out = attn(input_ids, batch_first=True)[0]
    except:
        return False

    return True


if __name__ == "__main__":
    seed()

    dim = 512
    head = 8
    batch_size = 4
    src_len = 100
    tgt_len = 50
    # dim = 4
    # head = 1
    # batch_size = 1
    # src_len = 100
    # tgt_len = 50
    # causal=False and cross=False are default settings
    # assign causal and cross attributes when necessary
    attn = LocalAttention(embed_dim=dim, num_heads=head, causal=False)
    print("Self-noncausal:", test_self_noncausal(attn))

    attn = LocalAttention(embed_dim=dim, num_heads=head, causal=True)
    print("Self-causal-incremental:", test_self_causal_incremental(attn))
    
    attn = LocalAttention(embed_dim=dim, num_heads=head, causal=True)
    print("Self-causal:", test_self_causal(attn))

    attn = LocalAttention(embed_dim=dim, num_heads=head, cross=True)
    print("Cross-noncausal:", test_cross_noncausal(attn))

    attn = LocalAttention(embed_dim=dim, num_heads=head, causal=True, cross=True)
    print("Cross-causal:", test_cross_causal(attn))
