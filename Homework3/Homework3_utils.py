import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdb
class NeuralNetwork():
    def __init__(self, layers):
        self.parameters, self.velocity, self.training = {}, {}, True
        self.layer_number = len(layers)
        for l in range(1, self.layer_number): # generate weight and bias according to the input layers
            self.parameters["W" + str(l)] = np.random.random((layers[l - 1], layers[l])) - 0.5
            self.parameters["b" + str(l)] = np.random.random(layers[l]) - 0.5

    def forward(self, X): # do forward propagation
        self.cache = {}
        input = X
        for l in range(1, self.layer_number):
            if self.training:
                self.cache["l" + str(l)] = input # save the input as cache for backpropagation
            output = np.dot(input, self.parameters["W" + str(l)]) + self.parameters["b" + str(l)] # input * weight + bias
            if l < self.layer_number - 1:
                input = sigmoid(output)
                
        return output
    
    def criterion(self, y_pd, y_gt): # calculate the loss and gradients
        loss = cross_entropy(softmax(y_pd), one_hot(y_gt)) # get cross entropy loss
        back = (softmax(y_pd) - one_hot(y_gt)) / y_gt.shape
        self.gradients = {}
        if self.training: #  only calculate gradient in training mode
            for l in reversed(range(1, self.layer_number)):
                self.gradients["dw" + str(l)] = np.dot(self.cache["l" + str(l)].T, back)
                self.gradients["db" + str(l)] = np.sum(back, axis=0)
                back = sigmoid_delta(self.cache["l" + str(l)]) * np.dot(back, self.parameters["W" + str(l)].T)
        
        return loss.mean()

    def step(self, learning_rate, momentum=0.9): # optimize the parameters by minus the gradient
        for l in range(1, self.layer_number):
            try:
                self.velocity["Vw" + str(l)] = self.velocity["Vw" + str(l)] * momentum + self.gradients["dw" + str(l)]
                self.velocity["Vb" + str(l)] = self.velocity["Vb" + str(l)] * momentum + self.gradients["db" + str(l)]
            except: # there is no velocity in the first time, so set the gradients as velocity
                self.velocity["Vw" + str(l)] = self.gradients["dw" + str(l)]
                self.velocity["Vb" + str(l)] = self.gradients["db" + str(l)]
            # update the parameters using velocity
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * self.velocity["Vw" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * self.velocity["Vb" + str(l)]

    def max(self, output): # convert the output to predicted class
        predict = []
        for o in output:
            p, = np.where(o == o.max())
            predict.append(p[0])
        
        return predict

    def no_grad(self): # clear the gradients and cache
        self.cache = {}
        self.gradients = {}

    def eval(self): # turn to evaluation mode
        self.training = False

    def train(self): # turn to train mode
        self.training = True

def get_data(batchsize, preprocess=False): # get datas from path then return
    train_dir, test_dir = "./Data/Data_train", "./Data/Data_test"
    classes = ["Carambula", "Lychee", "Pear"]
    train_data, test_data = [], []
    for i, clas in enumerate(classes):
        trainset = [(np.array(Image.open(os.path.join(train_dir, clas, p))), i) for p in os.listdir(os.path.join(train_dir, clas))]
        testset = [(np.array(Image.open(os.path.join(test_dir, clas, p))), i) for p in os.listdir(os.path.join(test_dir, clas))]
        train_data.extend(trainset), test_data.extend(testset)
    if preprocess:
        train_x, train_y = pre_process(train_data, shuffle=True, normalize=True)
        test_x, test_y = pre_process(test_data, shuffle=True, normalize=True)
    else:
        train_x, train_y = pre_process(train_data, shuffle=False, normalize=False)
        test_x, test_y = pre_process(test_data, shuffle=False, normalize=False)
    
    return train_x, train_y, test_x, test_y

def pre_process(dataset, shuffle=True, normalize=True): # do pre process to data, e.g. shuffle, normalize
    x, y = [], []
    if shuffle:
        random.shuffle(dataset)
    for (a, b) in dataset:
        if normalize:
            a = a / 255 - 0.5
        x.append(a.reshape(-1)), y.append(b)

    return np.array(x), np.array(y)


def dataloader(x, y, batchsize): # load data as a dataloader
    length = x.shape[0]
    loader = []
    if length % batchsize == 0:
        for i in range(int(length/batchsize)):
            loader.append((x[i*batchsize:(i+1)*batchsize], y[i*batchsize:(i+1)*batchsize]))
    else:
        for i in range(int(length/batchsize)):
            loader.append((x[i*batchsize:(i+1)*batchsize], y[i*batchsize:(i+1)*batchsize]))
        loader.append((x[(i+1)*batchsize:], y[(i+1)*batchsize:]))
    
    return loader

def softmax(y): # softmax function
    exp_y = np.exp(y)
    sum = np.sum(exp_y, axis=1)
    
    return exp_y / sum[:, np.newaxis]

def one_hot(y): # one hot encoder for y
    out_y = np.zeros((y.shape[0], 3))
    out_y[np.arange(y.shape[0]), y] = 1
    
    return out_y

def cross_entropy(y_pd, y_gt, epsilon=1e-10): # calculate cross entropy loss
    loss = -y_gt * np.log10(y_pd + epsilon)
    
    return loss

def sigmoid(x): # sigmoid function
    return 1 / (1 + np.exp(-x))

def sigmoid_delta(x): # derivative of relu function
    f = sigmoid(x)
    return f * (1 - f)

def plot_decision_region(x, y, filename):
    plt.figure(figsize=(10,6))
    plt.grid(True)
    for input, target in zip(x, y):
        if target == 0:
            plt.scatter(input[0], input[1], c='r', marker='o', s=5)
        elif target == 1:
            plt.scatter(input[0], input[1], c='g', marker='o', s=5)
        elif target == 2:
            plt.scatter(input[0], input[1], c='b', marker='o', s=5)
    red_patch = mpatches.Patch(color='red', label="Carambula")
    green_patch = mpatches.Patch(color='green', label="Lychee")
    blue_patch = mpatches.Patch(color='blue', label="Pear")
    plt.legend(handles=[red_patch, green_patch, blue_patch])
    plt.savefig(filename + ".png")

def plot_loss_acc(loss, acc, e, name, part):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(e, loss)
    axs[0].set_title(name + " Loss")
    axs[0].set_xlabel("epoch")
    axs[1].plot(e, acc)
    axs[1].set_title(name + " Accuracy")
    axs[1].set_xlabel("epoch")
    axs[0].grid(), axs[1].grid()
    fig.savefig(name + "_" + part + ".png")