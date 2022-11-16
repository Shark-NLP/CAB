import torch
from collections import OrderedDict
from typing import List
import sys,os

ckpt2average = ['I1000000_E58_gen.pth', 'I950000_E55_gen.pth', 'I900000_E52_gen.pth', 'I850000_E49_gen.pth', 'I800000_E46_gen.pth']


def get_avg_ckpt(p, device='cpu'):
    ckpt_paths = [os.path.join(p, i) for i in os.listdir(p)]
    state_dict_list = []
    for path in ckpt_paths:
        if os.path.split(path)[-1] in ckpt2average:
            with open(path, 'rb') as fin:
                state_dict_list.append(torch.load(fin, map_location='cpu'))
    state_dict = average_checkpoints(state_dict_list)
    if device != 'cpu':
        state_dict = {k: v.to(device) for k, v in state_dict.items()}
    torch.save(state_dict, os.path.join(p, 'Average_gen.pth'))


def average_checkpoints(state_dict_list: List):
    state_dict = OrderedDict()
    for i, sd in enumerate(state_dict_list):
        for key in sd:
            p = sd[key]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if i == 0:
                state_dict[key] = p.numpy()
            else:
                state_dict[key] = state_dict[key] + p.numpy()
    ckpt_num = len(state_dict_list)
    for key in state_dict:
        state_dict[key] = state_dict[key] / ckpt_num
        state_dict[key] = torch.from_numpy(state_dict[key])
    return state_dict


if __name__ == '__main__':
    get_avg_ckpt(sys.argv[1])
