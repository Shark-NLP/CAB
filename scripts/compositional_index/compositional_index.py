import sys
import numpy as np

mu = [[3.4245, 1.97429, 4.06947, 2.19021, 34.88667, 6.79333, 31.96889, 23.20800, 0.67580, 36.29200],
      [4.08325, 2.20085, 32.95750, 6.09250, 30.25750, 22.26000],
      [7.70250, 0.24775, 0.76250, 1.18933, 0.80625, 0.47575, 0.51100, 0.62025, 0.58075],
      [5.50370, 2.62780, 31.35000, 5.26000, 28.80000 ]]
sigma = [[0.06832, 0.0361768, 0.09422, 0.03866, 1.39697, 1.41078, 1.31499, 0.27499, 0.01557, 62.30827],
         [0.04529, 0.01372, 1.92754, 0.41007, 1.61951, 8.29944],
         [0.64789, 0.03366, 0.03906, 0.05615, 0.02193, 0.01057, 0.00601, 0.18948, 0.10312 ],
         [1.29239, 0.42726, 3.77103, 1.26028, 3.28638]]
metric_weight = [[-1, -1, -1, -1, 1, 1, 1, 1, 1, -1],
                 [-1, -1, 1, 1, 1, -1],
                 [-1, -1, 1, -1, -1, -1, -1, -1, -1],
                 [-1, -1, 1, 1, 1]]

pattern = ['Noncausal Self', 'Causal Self', 'Noncausal Cross', 'Causal Cross']
task_average = {'Noncausal Self': [[0, 1, 2, 3], [4, 5, 6], [7, 8], [9]],
                'Causal Self': [[0, 1], [2, 3, 4], [5]],
                'Noncausal Cross': [[0, 1, 2], [3, 4, 5, 6, 7, 8]],
                'Causal Cross': [[0, 1], [2, 3, 4]]}
task_name = {'Noncausal Self': ['TTS', 'Sum', 'SR', 'MLM'],
             'Causal Self': ['TTS', 'Sum', 'LM'],
             'Noncausal Cross': ['PCC', 'LSTF'],
             'Causal Cross': ['TTS', 'Sum']}


def get_ci(path):
    with open(path, encoding='utf-8') as f:
        x = f.readlines()
    x = [list(map(float, line.split())) for line in x]

    for i in range(4):
        x_ = np.array(x[i])
        mu_ = np.array(mu[i])
        sigma_ = np.array(sigma[i])
        weight_ = np.array(metric_weight[i])
        score = (x_ - mu_) / sigma_ * weight_
        task_index = task_average[pattern[i]]
        task_score = [np.average(score[j]) for j in task_index]
        print(f"========{pattern[i]}========")
        for task, score in zip(task_name[pattern[i]], task_score):
            print(f"{task}: " + "{:.4f}".format(score), end='  ')
        print("\ntotal: {:.4f}".format(np.average(task_score)))


if __name__ == '__main__':
    get_ci(sys.argv[1])



