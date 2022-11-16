import math
import sys
from time import sleep, time

import numpy as np
import torch
from efficient_attention import MultiheadAttention, ABC
import pandas as pd


def build_self_attn(attention_type, max_seq_len=8192, causal=False):
    if attention_type == 'vanilla':
        return MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
        )
    elif attention_type == 'abc':
        return ABC(
            embed_dim=embed_dim,
            num_landmarks=landmarks,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal
        )
    return None


def get_efficiency(attn_name='vanilla', sequence_length=[256]):
    causal = False

    cases = [torch.randn(x, 4, embed_dim).cuda() for x in sequence_length]
    inference_time_list = []
    now_memory = torch.cuda.memory_stats()['active_bytes.all.current']>>20

    time_cost = [attn_name]
    memory_cost = [attn_name]
    for i, length in enumerate(sequence_length):
        model = build_self_attn(attn_name, causal=causal, max_seq_len=length).cuda()
        repeat = 100
        Q = cases[i].clone()
        latency = []
        inference_begin = time()
        for _ in range(repeat):
            begin = time()
            x, _ = model(Q, Q, Q)
            end = time()
            latency.append((end - begin) * 1000)
        inference_end = time()
        latency = np.array(latency)
        
        least, mid, most = np.percentile(latency, (25, 50, 75), interpolation='midpoint')
        latency[latency < least] = 0
        latency[latency > most] = 0
        nonzero_num = np.sum(latency != 0)
        print("{} takes {:.3f} miliseconds to run inference on {} length".format(attn_name, np.sum(latency) / nonzero_num, length))
        inference_time_list.append(str((inference_end - inference_begin) / repeat * 1000))
        print("peak memory usage (MB): {}".format((torch.cuda.memory_stats()['active_bytes.all.peak']>>20)-now_memory))
        time_cost.append(np.sum(latency) / nonzero_num)
        memory_cost.append(str((torch.cuda.memory_stats()['active_bytes.all.peak']>>20)-now_memory))
        sleep(1)

    return time_cost, memory_cost


if __name__ == "__main__":
    attn_name = sys.argv[1]
    embed_dim = 512
    num_heads = 8
    dropout = 0.3
    landmarks = 16
    wsize = 16
    conv_kernel_size = 5
    sequence_length = [256, 512, 1024, 2048, 4096, 8192]

    v_time, v_memory = get_efficiency(attn_name, sequence_length)

    columns = [''] + list(map(str, sequence_length))
    time_output = pd.DataFrame(np.array([v_time]), columns=columns)
    memory_output = pd.DataFrame(np.array([v_memory]), columns=columns)

    try:
        df: pd.DataFrame = pd.read_excel("cost.xlsx", 'time', index=False)
        df2: pd.DataFrame = pd.read_excel("cost.xlsx", 'memory', index=False)
        df.columns = columns
        df2.columns = columns

        df = df.append(time_output, ignore_index=True)
        df.to_excel("cost.xlsx", index=False, sheet_name="time")

        df2 = df2.append(memory_output, ignore_index=True)
        writer = pd.ExcelWriter("cost.xlsx", mode="a", engine="openpyxl")
        df2.to_excel(writer, index=False, sheet_name="memory")
        writer.save()
        writer.close()
    except:
        time_output.to_excel("cost.xlsx", index=False, sheet_name="time")
        writer = pd.ExcelWriter("cost.xlsx", mode="a", engine="openpyxl")
        memory_output.to_excel(writer, index=False, sheet_name="memory")
        writer.save()
        writer.close()
