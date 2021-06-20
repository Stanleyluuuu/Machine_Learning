import numpy as np
import random
import os
from PIL import Image

class NeuralNetwork():
    def __init__(self, layers):
        self.parameters, self.velocity, self.training = {}, {}, True
        self.layer_number = len(layers)
        for l in range(1, len(layers)): # generate weight and bias according to the input layers
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
                input = relu(output)
                
        return output

    # def batch_norm(self, x, l):
    #     mean, var, epsilon = x.mean(axis=1), x.var(axis=1), np.random.random(x.shape[0])
    #     self.x_hat["layer" + str(l)] = (x - mean[:, None]) / np.sqrt(var - epsilon)[:, None]
        
    #     return self.bn["gamma" + str(l)] * self.x_hat["layer" + str(l)] + self.bn["beta" + str(l)]
    
    def criterion(self, y_pd, y_gt): # calculate the loss and gradients
        loss = cross_entropy(softmax(y_pd), one_hot(y_gt)) # get cross entropy loss
        back = (softmax(y_pd) - one_hot(y_gt)) / y_gt.shape
        self.gradients = {}
        if self.training: #  only calculate gradient in training mode
            for l in reversed(range(1, self.layer_number)):
                self.gradients["dw" + str(l)] = np.dot(self.cache["l" + str(l)].T, back)
                self.gradients["db" + str(l)] = np.sum(back, axis=0)
                back = relu_delta(self.cache["l" + str(l)]) * np.dot(back, self.parameters["W" + str(l)].T)
            
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

def get_data(batchsize, shuffle=False, normalize=False): # get datas from path then return as a dataloader
    train_dir, test_dir = "./Data/Data_train", "./Data/Data_test"
    classes = ["Carambula", "Lychee", "Pear"]
    train_data, test_data = [], []
    for i, clas in enumerate(classes):
        trainset = [(np.array(Image.open(os.path.join(train_dir, clas, p))), i) for p in os.listdir(os.path.join(train_dir, clas))]
        testset = [(np.array(Image.open(os.path.join(test_dir, clas, p))), i) for p in os.listdir(os.path.join(test_dir, clas))]
        train_data.extend(trainset), test_data.extend(testset)
    train_x, train_y = pre_process(train_data, shuffle=True, normalize=True)
    test_x, test_y = pre_process(test_data, shuffle=True, normalize=True)
    trainloader = dataloader(train_x, train_y, batchsize=batchsize)
    testloader = dataloader(test_x, test_y, batchsize=batchsize)

    return trainloader, testloader

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

def relu(x): # relu function
    return np.maximum(0, x)

def relu_delta(x): # derivative of relu function
    return (x > 0).astype(np.float)