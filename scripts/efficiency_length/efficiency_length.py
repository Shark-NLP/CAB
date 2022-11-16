import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
try:
    import openpyxl, xlrd
except:
    print("Please install openpyxl and xlrd package: pip install openpyxl xlrd==1.2.0")
    exit(-1)
    
    
memory_df: pd.DataFrame = pd.read_excel("./efficiency_length/cost.xlsx", 'memory')
seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192]).reshape(-1, 1)
mha_memory = memory_df.iloc[0, 1:]
attn_num = memory_df.shape[0]
quadratic_featurizer = PolynomialFeatures(2)
seq_transform = quadratic_featurizer.fit_transform(seq_lengths)

mha_model = LinearRegression().fit(seq_transform, mha_memory)
mha_coef_ = mha_model.coef_
mha_intercept_ = mha_model.intercept_
a = mha_coef_[2]
b = mha_coef_[1]
c = mha_intercept_
R2 = mha_model.score(seq_transform, mha_memory)
memory_df['intercept_point'] = 0
memory_df['R^2'] = R2
memory_df['a'] = a
memory_df['b'] = b
memory_df['c'] = c
print("===========Memory=========")
for i in range(1, attn_num):
    memory = memory_df.iloc[i, 1:-5]
    model = LinearRegression().fit(seq_lengths, memory)
    coef = model.coef_
    intercept = model.intercept_
    R2 = model.score(seq_lengths, memory)
    print(f"{memory_df.iloc[i, 0]}")
    d = coef
    e = intercept
    intersect_point = (-(b - d) + np.sqrt((b - d) ** 2 - 4 * a * (c - e))) / (2 * a)
    intersect_point2 = (-(b - d) - np.sqrt((b - d) ** 2 - 4 * a * (c - e))) / (2 * a)
    print(f"Efficiency Length: {int(round(intersect_point[0]))}, R^2: {R2}")
    memory_df.iloc[i, 7] = intersect_point[0]

memory_df: pd.DataFrame = pd.read_excel("./efficiency_length/cost.xlsx", 'time')
seq_lengths = np.array([256, 512, 1024, 2048, 4096, 8192]).reshape(-1, 1)
mha_memory = memory_df.iloc[0, 1:]
quadratic_featurizer = PolynomialFeatures(2)
seq_transform = quadratic_featurizer.fit_transform(seq_lengths)

mha_model = LinearRegression().fit(seq_transform, mha_memory * 1000)
mha_coef_ = mha_model.coef_
mha_intercept_ = mha_model.intercept_
a = mha_coef_[2]
b = mha_coef_[1]
c = mha_intercept_
R2 = mha_model.score(seq_transform, mha_memory * 1000)
memory_df['intercept_point'] = 0
memory_df['R^2'] = R2
memory_df['a'] = a
memory_df['b'] = b
memory_df['c'] = c
print("===========Time=========")
for i in range(1, attn_num):
    memory = memory_df.iloc[i, 1:-5]
    model = LinearRegression().fit(seq_lengths[1:], memory[1:] * 1000)
    coef = model.coef_
    intercept = model.intercept_
    R2 = model.score(seq_lengths[1:], memory[1:] * 1000)
    print(f"{memory_df.iloc[i, 0]}")
    d = coef
    e = intercept
    intersect_point = (-(b - d) + np.sqrt((b - d) ** 2 - 4 * a * (c - e))) / (2 * a)
    intersect_point2 = (-(b - d) - np.sqrt((b - d) ** 2 - 4 * a * (c - e))) / (2 * a)
    print(f"Efficiency Length: {int(round(intersect_point[0]))}, R^2: {R2}")
    memory_df.iloc[i, 7] = intersect_point[0]
