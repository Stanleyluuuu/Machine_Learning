import numpy as np
import csv
import torch
import pdb
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 3, stride=1)
        self.conv2 = nn.Conv1d(6, 12, 3, stride=1)
        self.conv3 = nn.Conv1d(12, 12, 3, stride=1)
        self.conv4 = nn.Conv1d(12, 6, 3, stride=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(6)
        self.bn2 = nn.BatchNorm1d(12)
        self.fc1 = nn.Linear(6 * 16,  150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.bn1(F.relu(self.conv4(x)))
        x = x.view(-1, 6 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

def read_file(data_dir, label=0, number=20621):
    data = []
    if label == 1:
        labels = []
    elif label == 5:
        # labels = np.zeros((20621, 5))
        labels = np.zeros((number, 5))

    with open(data_dir) as d:
        rows = csv.reader(d)
        for i, row in enumerate(rows):
            if i > 0:
                row = [float(a) for a in row[2:]]
                if label != 0:
                    data.append(row[:-1])
                else:
                    data.append(row)
                if label == 1:
                    labels.append(int(row[-1]) - 1)
                elif label == 5:
                    labels[i-1][int(row[-1]) - 1] = 1
    if label == 0:
        return np.array(data)
    else:
        return np.array(data), np.array(labels)


def get_data(args, row=5): # get the datas from csv file and turns into the format I need
    # train_dir, test_dir = './fgd_data/train.csv', './fgd_data/test.csv'
    train_dir, val_dir, test_dir = 'train.csv', 'val.csv', './fgd_data/test.csv'
    train_x, train_y = read_file(train_dir, row, number=19000)
    val_x, val_y = read_file(val_dir, row, number=1621)
    test = read_file(test_dir)
    if args.method == "CNN":
        # ex_tx, ex_t = train_x[:, np.newaxis, :], test[:, np.newaxis, :]
        ex_tx, ex_v, ex_t = train_x[:, np.newaxis, :], val_x[:, np.newaxis, :], test[:, np.newaxis, :]
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        t_train_x, t_train_y, t_test = np.einsum('abc->bca', preprocess(ex_tx)), torch.tensor(train_y, dtype=torch.long), np.einsum('abc->bca', preprocess(ex_t))
        t_val_x, t_val_y = np.einsum('abc->bca', preprocess(ex_v)), torch.tensor(val_y, dtype=torch.long)
        trainset = torch.utils.data.TensorDataset(torch.Tensor(t_train_x), t_train_y)
        valset = torch.utils.data.TensorDataset(torch.Tensor(t_val_x), t_val_y)
        testset = torch.utils.data.TensorDataset(torch.Tensor(t_test))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False)
        return trainloader, valloader, testloader
    else:
        
        return train_x, train_y, val_x, val_y, test


def softmax(prediction): # softmax function
    a = np.exp(prediction)
    b = a.sum(axis=1)

    return a / b[:, np.newaxis]

def get_loss(prediction, y, epsilon=1e-10): # calculate cross entropy loss, adding epsilon to make sure no log(0) won't appear
    log_loss = (1 / prediction.shape[0]) * np.nansum(y * np.log(prediction + epsilon))
    
    return -log_loss

def save(prediction, filename): # save the result as test.csv file in the format of AIdea
    with open(filename + ".csv", 'w') as f:
        f.write("ID,C1,C2,C3,C4,C5")
        f.write("\n")
        for i, p in enumerate(prediction):
            f.write(str(i+1))
            for n in p:
                f.write("," + str(n))
            if i != 499:
                f.write("\n")

def split(): # split some data from train.csv as validation set
    train_dir = './fgd_data/train.csv'
    with open(train_dir, 'r') as f:
        a = f.readlines()
    zz, train, val = [], [], []
    for i, d in enumerate(a):
        if i > 0:
            zz.append(d[:-1 ]if d[-1:] == "\n" else d)
    random.shuffle(zz)
    for i, f in enumerate(zz):
        if i > 1620:
            train.append(f)
        else:
            val.append(f)
    with open("train.csv", "w") as f:
        f.write(a[0])
        for i, d in enumerate(train):
            f.write(d)
            if i == len(train) - 1:
                pass
            else:
                f.write("\n")
    with open("val.csv", "w") as f:
        f.write(a[0])
        for i, d in enumerate(val):
            f.write(d)
            if i == len(val) - 1:
                pass
            else:
                f.write("\n")
    pdb.set_trace()
