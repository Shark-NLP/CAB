import os
from collections import defaultdict
import numpy as np
import sys


def get_answer(path):
    results_mae = defaultdict(float)
    results_mse = defaultdict(float)
    results_mae_m = defaultdict(float)
    results_mse_m = defaultdict(float)
    itr = defaultdict(float)
    for ls in os.listdir(path):
        o = ls.split('_')
        for file in os.listdir(os.path.join(path, ls)):
            if file != 'metrics.npy': continue
            ans = np.load(os.path.join(path, ls, file))
            results_mae['_'.join([o[1],o[2][2:],o[5][2:],o[11][2:]])] += ans[0]
            results_mse['_'.join([o[1],o[2][2:],o[5][2:],o[11][2:]])] += ans[1]
            if results_mae_m['_'.join([o[1],o[2][2:],o[5][2:],o[11][2:]])] > 0:
                results_mae_m['_'.join([o[1], o[2][2:], o[5][2:], o[11][2:]])] = min(results_mae_m['_'.join([o[1],o[2][2:],o[5][2:],o[11][2:]])], ans[0])
                results_mse_m['_'.join([o[1], o[2][2:], o[5][2:], o[11][2:]])] = min(results_mse_m['_'.join([o[1], o[2][2:], o[5][2:], o[11][2:]])], ans[1])
            else:
                results_mae_m['_'.join([o[1], o[2][2:], o[5][2:], o[11][2:]])] = ans[0]
                results_mse_m['_'.join([o[1], o[2][2:], o[5][2:], o[11][2:]])] = ans[1]
            itr['_'.join([o[1],o[2][2:],o[5][2:],o[11][2:]])] += 1

    for k in results_mae:
        results_mae[k] = round(results_mae[k]/itr[k], 3)
        results_mse[k] = round(results_mse[k]/itr[k], 3)
        print(f"======{k}======")
        print(f"MSE: {results_mse[k]} MAE: {results_mae[k]}")
        print(f"Min_MSE: {round(results_mse_m[k].astype(np.float64), 3)} Min_MAE: {round(results_mae_m[k].astype(np.float64), 3)}")


if __name__ == '__main__':
    get_answer(sys.argv[1])

