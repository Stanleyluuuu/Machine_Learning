import torch
import torchvision
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pdb

data_dir = './fgd_data/train.csv'
datas = [d[:-1] for d in open(data_dir)]
datas_split = [b.split(",") for b in datas]
pdb.set_trace()