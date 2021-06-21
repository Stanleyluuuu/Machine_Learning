from Homework3_utils import *
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pdb
'''
To reproduce my work, you have to put "Data" folder in the same folder as these two python file.
There are "Data_train" and "Data_test" inside "Data", and there three classes inside
each training and testing folder.

python Homework3.py --batchsize 200 --lr 0.001 --momentum 0.9 --epoch 500 --p1
'''
parser = argparse.ArgumentParser(description='Machine Learning Homework 3')
parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                    help='Input batch size for training (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='Learning rate for updating parameters (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Momentum for using SGD (default: 0.9)')
parser.add_argument('--epoch', type=int, default=200, metavar='E',
                    help='Number of training epoch (default: 200)')
parser.add_argument('--p1', action='store_true', default=False,
                    help='Whether to run part 1')
parser.add_argument('--p2', action='store_true', default=False,
                    help='Whether to run part 2')
args = parser.parse_args()

if args.p1:
    layers = [3, 1200, 3] # define model layers, first layer must be 3(input size), and last layer must be 3(classification head)
    print("Start Part 1")
if args.p2:
    layers = [3, 200, 150, 3] # define model layers, first layer must be 3(input size), and last layer must be 3(classification head)
    print("Start Part 2")
train_x, train_y, test_x, test_y = get_data(batchsize=args.batchsize, preprocess=True) # get data from the folder with preprocess, i.e. normalize, shuffle
pca = PCA(n_components=2)
pca.fit(train_x)
pca_train_x, pca_test_x = pca.transform(train_x), pca.transform(test_x) # transform data into 2 dimensions
b1, b2 = np.random.random(pca_train_x.shape[0]), np.random.random(pca_test_x.shape[0]) # create bias
x_train, x_test = np.hstack((pca_train_x, b1[:, np.newaxis])), np.hstack((pca_test_x, b2[:, np.newaxis]))
trainloader = dataloader(x_train, train_y, batchsize=args.batchsize)
testloader = dataloader(x_test, test_y, batchsize=args.batchsize)
model = NeuralNetwork(layers)
train_loss, train_acc, test_loss, test_acc = [], [], [], []
print("Start training")
for epoch in range(args.epoch):
    epoch_loss, count, acc = 0, 0, 0
    model.train()
    print("Epoch number {e1}".format(e1=epoch+1))
    for data in trainloader:
        inputs, labels = data[0], data[1] # get images and labels from the dataloader
        model.no_grad() # clean up the cache and the gradients of last batch
        output = model.forward(inputs) # input the images into the model and get the result after going through softmax, which is predicted probability
        predicted = model.max(output) # convert the output probability into predicted classes
        acc += (predicted == labels).sum() / labels.shape[0] # calculate the accuracy
        loss = model.criterion(output, labels) # calculate the cross entropy loss
        epoch_loss += loss
        count += 1
        model.step(args.lr, momentum=args.momentum) # update the parameters using gradient and momentum
    train_loss.append(epoch_loss/count), train_acc.append(acc/count)
    print("Training loss = {e1:5f}, accuracy = {e2:3f}".format(e1=epoch_loss/count, e2=acc/count))
    model.eval()
    epoch_loss, count, acc = 0, 0, 0
    for data in testloader:
        inputs, labels = data[0], data[1] # get images and labels from the dataloader
        output = model.forward(inputs) # input the images into the model and get the result after going through softmax, which is predicted probability
        predicted = model.max(output) # convert the output probability into predicted classes
        acc += (predicted == labels).sum() / labels.shape[0] # calculate the accuracy
        loss = model.criterion(output, labels) # calculate the cross entropy loss
        epoch_loss += loss
        count += 1
    test_loss.append(epoch_loss/count), test_acc.append(acc/count)
    print("Testing loss = {e1:5f}, accuracy = {e2:3f}".format(e1=epoch_loss/count, e2=acc/count))
    print("##########################################################################")
print("###################")
print("# Finish training #")
print("###################")
# output_train = model.forward(x_train)
# output_test = model.forward(x_test)
# predicted_train = model.max(output_train)
# predicted_test = model.max(output_test)
predicted_train = model.max(model.forward(x_train))
predicted_test = model.max(model.forward(x_test))
e = np.linspace(1, args.epoch, args.epoch)
print("Start plotting")
if args.p1:
    plot_decision_region(x_train, predicted_train, "decision_region_p1_train")
    plot_decision_region(x_test, predicted_test, "decision_region_p1_test")
    plot_loss_acc(train_loss, train_acc, e, "Training", "part1")
    plot_loss_acc(test_loss, test_acc, e, "Testing", "part1")
if args.p2:
    plot_decision_region(x_train, predicted_train, "decision_region_p2_train")
    plot_decision_region(x_test, predicted_test, "decision_region_p2_test")
    plot_loss_acc(train_loss, train_acc, e, "Training", "part2")
    plot_loss_acc(test_loss, test_acc, e, "Testing", "part2")
print("Finish plotting")