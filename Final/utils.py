import numpy as np
import pandas as pd
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv1d(1, 12, 3, stride=1)
        self.conv2 = nn.Conv1d(12, 12, 3, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(12)
        self.fc1 = nn.Linear(12 * 4, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 300)
        self.fc4 = nn.Linear(300, 5)
        self.fcbn1 = nn.BatchNorm1d(250)
        self.fcbn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 12 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)

        return x

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv1d(1, 12, 3, stride=1)
        self.conv2 = nn.Conv1d(12, 12, 3, stride=1)
        self.conv3 = nn.Conv1d(12, 12, 3, stride=1)
        self.conv4 = nn.Conv1d(12, 12, 3, stride=1)
        self.conv5 = nn.Conv1d(12, 12, 3, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(12)
        self.bn3 = nn.BatchNorm1d(12)
        self.bn4 = nn.BatchNorm1d(12)
        self.bn5 = nn.BatchNorm1d(12)
        self.fc1 = nn.Linear(12 * 15, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 300)
        self.fc4 = nn.Linear(300, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.bn4(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 12 * 15)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)

        return x

class ListDataset(object):
    def __init__(self):
        pass

    def load_data(self, file_dir, lb=None):
        df = pd.read_csv(file_dir)
        df = pd.DataFrame(data=df)

        for i in range(len(df)): # encode the time and date
            date, time = df['TS'][i].split(' ')
            date = date.split('/')
            if int(date[2])<10: # encoding date
                date = date[1]+str(0)+date[2]
            else:
                date = date[1]+date[2]
            time = time.split(':')
            if int(time[0])<10: # encoding time
                time = str(0)+time[0]+time[1]
            else:
                time = time[0]+time[1]
            df['TS'][i] = date+time
        df = df.apply(pd.to_numeric, errors='coerce')
        attributes = ['TS', 'X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08', 'X09', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24']
        x = df[attributes]
        if lb:
            y = df['Y']

            return np.array(x), np.array(y)
        else:

            return np.array(x)

    def get_item(self, file_dir, lb=None):
        if lb:
            data, labels = self.load_data(file_dir, lb)
        else:
            data = self.load_data(file_dir, lb)
        pca = PCA(n_components=1)
        d_01 = data[:, 0]
        p1 = np.expand_dims(d_01, axis=1)
        d_02 = data[:, 1]
        p2 = np.expand_dims(d_02, axis=1)
        d_03to08 = data[:, 2:8]
        p3 = pca.fit(d_03to08).transform(d_03to08)
        d_09to010 = data[:, 8:10]
        p4 = pca.fit(d_09to010).transform(d_09to010)
        d_11to12 = data[:, 10:12]
        p5 = pca.fit(d_11to12).transform(d_11to12)
        d_13 = data[:, 12:13]
        d_22to25 = data[:, 21:]
        g1 = np.concatenate((d_13, d_22to25), axis=1)
        p6 = pca.fit(g1).transform(g1)
        d_14to16 = data[:, 13:16]
        d_17 = data[:, 16:17]
        d_18 = data[:, 17:18]
        d_21 = data[:, 20:21]
        g2 = np.concatenate((d_14to16, d_17, d_18, d_21), axis=1)
        p7 = pca.fit(g2).transform(g2)
        d_19to20 = data[:, 17:19]
        p8 = pca.fit(d_19to20).transform(d_19to20)
        group_data = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8), axis=1)
        if lb:
            x_train ,x_val ,y_train, y_val = train_test_split(group_data, labels, test_size=0.2, random_state=0)
        
            return  x_train ,x_val ,y_train, y_val
        else:

            return group_data

def get_data(file_dir, train=False):
    df = pd.read_csv(file_dir)
    df = pd.DataFrame(data=df)

    for i in range(len(df)): # encode the time and date
        date, time = df['TS'][i].split(' ')
        date = date.split('/')
        if int(date[2])<10: # encoding date
            date = date[1]+str(0)+date[2]
        else:
            date = date[1]+date[2]
        time = time.split(':')
        if int(time[0])<10: # encoding time
            time = str(0)+time[0]+time[1]
        else:
            time = time[0]+time[1]
        df['TS'][i] = date+time
    df = df.apply(pd.to_numeric, errors='coerce')
    attributes = ['TS', 'X01', 'X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08', 'X09', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24']
    x = df[attributes]
    if train:
        y = df['Y']
        x_train ,x_val ,y_train, y_val = train_test_split(x,y, test_size=0.2, random_state=0)
        
        return np.array(x_train), np.array(y_train).astype(float), np.array(x_val), np.array(y_val).astype(float)
    else:
        return np.array(x)

def save(x, method):
    # pdb.set_trace()
    predicted = pd.DataFrame(x)
    predicted.columns = ['C1','C2','C3','C4','C5']
    predicted.insert(loc=0, column='ID', value=np.arange(500)+1)
    name = method + ".csv"
    predicted.to_csv(name, index=False)

def one_hot(y): # one hot encoder for y
    out_y = np.zeros((y.shape[0], 5))
    out_y[np.arange(y.shape[0]), y] = 1
    
    return out_y

def get_loader(x, args, y=None):
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    tensor_x = np.einsum('abc->bac', preprocess(x))
    if len(y) == 0:
        dataset = torch.utils.data.TensorDataset(torch.Tensor(tensor_x))
        loader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=False)
    else:
        dataset = torch.utils.data.TensorDataset(torch.Tensor(tensor_x), torch.tensor(y - 1, dtype=torch.long))
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

    return loader

def softmax(y): # softmax function
    exp_y = np.exp(y)
    sum = np.sum(exp_y, axis=1)

    return exp_y / sum[:, np.newaxis]
    
    