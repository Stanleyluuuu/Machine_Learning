import torch
import torchvision
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pdb
import csv
import numpy as np
import matplotlib.pyplot as plt

data_dir = './fgd_data/train.csv'
train_x, train_y = [], []
with open(data_dir) as d:
    rows = csv.reader(d)
    for i, row in enumerate(rows):
        if i > 0:
            row = [float(a) for a in row[2:]]
            train_x.append(row[:-1])
            train_y.append(int(row[-1]))
train_x, train_y = np.array(train_x), np.array(train_y)
correlation = []
for i in range(24):
    correlation.append(np.corrcoef(train_x[:, i], train_y)[1][0])
left = np.arange(24) + 1
label = []
for n in left:
    name = 'X' + str(n)
    label.append(name)
plt.figure(figsize=(20, 9))
plt.bar(left, correlation, tick_label=label, width=0.8)
plt.xlabel('Different dimension of X')
plt.ylabel('Correlation')
plt.title('Correlation between different dimension of X')
plt.grid()
plt.savefig("GG.png")
