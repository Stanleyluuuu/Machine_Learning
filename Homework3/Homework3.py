from Homework3_utils import *
import argparse
import matplotlib.pyplot as plt
'''
In this homework, I implement a image classifier using SGD with momentum.
I use ReLU as activation function, and softmax at the last layer.
I didn't apply batch normalization in this work.

To reproduce my work, you have to put "Data" folder in the same folder as these two python file.
There are "Data_train" and "Data_test" inside "Data", and there three classes inside
each training and testing folder.

python Homework3.py --batchsize 200 --lr 0.001 --momentum 0.9 --epoch 300 --shuffle --normalize
'''
parser = argparse.ArgumentParser(description='Machine Learning Fianl Project')
parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                    help='Input batch size for training (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='Learning rate for updating parameters (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Momentum for using SGD (default: 0.9)')
parser.add_argument('--epoch', type=int, default=200, metavar='E',
                    help='Number of training epoch (default: 200)')
parser.add_argument('--shuffle', action='store_true', default=False,
                    help='Wether shuffle the dataset')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='Wether normalize the dataset')
args = parser.parse_args()
########################## Hyperparameters ##############################
layers = [2048, 800, 100, 3] # define model layers, first layer must be 2048(input size), and last layer must be 3(classification head)
#########################################################################
trainloader, testloader = get_data(batchsize=args.batchsize, shuffle=args.shuffle, normalize=args.normalize) # get datas as a dataloader
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
a = np.linspace(1, args.epoch, args.epoch)
fig, axs = plt.subplots(1, 2)
axs[0].plot(a, train_loss)
axs[0].set_title("Training Loss")
axs[0].set_xlabel("epoch")
axs[1].plot(a, train_acc)
axs[1].set_title("Training Accuracy")
axs[1].set_xlabel("epoch")
fig.savefig("Training.png")

fig2, axx = plt.subplots(1, 2)
axx[0].plot(a, test_loss)
axx[0].set_title("Testing Loss")
axx[0].set_xlabel("epoch")
axx[1].plot(a, test_acc)
axx[1].set_title("Testing Accuracy")
axx[1].set_xlabel("epoch")
fig2.savefig("Testing.png")