import argparse
import numpy as np
import os
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import *

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Learning Fianl Project')
    parser.add_argument('--method', type=str, default="CNN", metavar='M',
                        help='Input the method you want to use, e.g. LR(Linear Regression), KNN(K-Nearest Neighbor), DT(Decesion Tree), RF(Random Forest) and CNN1D, CNN2D(Convolutional Neural Network)(default: CNN)')
    parser.add_argument('--batchsize', type=int, default=200, metavar='N',
                        help='Input batch size for training (default: 100)')
    parser.add_argument('--epoch', type=int, default=500, metavar='N',
                        help='Input number of epoch for training (default: 500)')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='o',
                        help='Input the optimizer, Adam or SGD (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Input the learning rate of the optimizer (default: 0.001)')
    parser.add_argument('--model', type=int, default=1, metavar='Model',
                        help='Which model to use? 1 or 2 (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Use training mode')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Use testing mode')
    parser.add_argument('--n', type=int, default=500,
                        help='For testing')        
    args = parser.parse_args()
    # get the data
    train_dir, test_dir = './fgd_data/train.csv', './fgd_data/test.csv'
    if args.method == "LR":
        x_train, y_train, x_val, y_val = get_data(train_dir, train=True)
        x_test = get_data(test_dir)
        model = LinearRegression(normalize=True)
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        loss = metrics.mean_absolute_error(y_val, predictions)
        accuracy = metrics.accuracy_score(y_val, np.round(predictions, 0))
        print('\n\n============================================')
        print('Linear Regression\nAccuracy = {e1}\nMean absolute error = {e2}'.format(e1=accuracy, e2=loss))
        print('============================================\n\n')
        prediction_test = model.predict(x_test)
        save(one_hot(prediction_test.astype(int) - 1), "LR")
    if args.method == "KNN":
        x_train, y_train, x_val, y_val = get_data(train_dir, train=True)
        x_test = get_data(test_dir)
        model = KNeighborsClassifier(n_neighbors=50)
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        loss = metrics.mean_absolute_error(y_val, predictions)
        accuracy = metrics.accuracy_score(y_val, predictions)
        print('\n\n============================================')
        print('K Nearest Neighborhood\nAccuracy = {e1}\nMean absolute error = {e2}'.format(e1=accuracy, e2=loss))
        print('============================================\n\n')
        prediction_test = model.predict_proba(x_test)
        save(prediction_test, "KNN")
    if args.method == "DT":
        x_train, y_train, x_val, y_val = get_data(train_dir, train=True)
        x_test = get_data(test_dir)
        model = DecisionTreeClassifier(random_state=1, max_depth=30, min_samples_leaf=1, max_leaf_nodes=1500)
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        loss = metrics.mean_absolute_error(y_val, predictions)
        accuracy = metrics.accuracy_score(y_val, predictions)
        print('\n\n============================================')
        print('Decision Tree\nAccuracy = {e1}\nMean absolute error = {e2}'.format(e1=accuracy, e2=loss))
        print('============================================\n\n')
        prediction_test = model.predict_proba(x_test)
        save(prediction_test, "DT")
    if args.method == "RF":
        x_train, y_train, x_val, y_val = get_data(train_dir, train=True)
        x_test = get_data(test_dir)
        model = RandomForestClassifier(random_state=1, n_estimators=200, max_depth=30, min_samples_leaf=1, n_jobs=-1)
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        loss = metrics.mean_absolute_error(y_val, predictions)
        accuracy = metrics.accuracy_score(y_val, predictions)
        print('\n\n============================================')
        print('Random Forest\nAccuracy = {e1}\nMean absolute error = {e2}'.format(e1=accuracy, e2=loss))
        print('============================================\n\n')
        prediction_test = model.predict_proba(x_test)
        save(prediction_test, "RF")
    if args.method == "CNN":
        if args.model == 1:
            x_train, y_train, x_val, y_val = get_data(train_dir, train=True)
            x_test = get_data(test_dir)
            trainloader = get_loader(x_train, args, y_train)
            valloader = get_loader(x_val, args, y_val)
            testloader = get_loader(x_test, args, y=[])
            model = CNN1()
        elif args.model == 2:
            datalist = ListDataset()
            x_train ,x_val ,y_train, y_val = datalist.get_item(train_dir, lb=True)
            x_test = datalist.get_item(test_dir, lb=False)
            trainloader = get_loader(x_train, args, y_train)
            valloader = get_loader(x_val, args, y_val)
            testloader = get_loader(x_test, args, y=[])
            model = CNN2()
        if args.train:
            sweep_config = {'method': 'random', #grid, random
                            'metric': {'name': 'loss',
                                       'goal': 'minimize'},
                            'parameters': {'epochs': {'values': [args.epoch]},
                                           'batch_size': {'values': [args.batchsize]},
                                           'learning_rate': {'values': [args.lr]},
                                           'optimizer': {'values': ['adam', 'sgd']}}}
            sweep_id = wandb.sweep(sweep_config, project="Machine Learning Final Project")
            config_defaults = {'epochs': args.epoch,
                               'batch_size': args.batchsize,
                               'learning_rate': args.lr,
                               'optimizer': args.optimizer}
            wandb.init(config=config_defaults)
            config = wandb.config
            print("Training the Convolution Neural Network model")
            if args.optimizer == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            elif args.optimizer == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
            w = torch.tensor((0.03929472, 0.03407837, 0.07255861, 0.26796699, 0.5861013))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            # criterion = nn.CrossEntropyLoss(weight=w.to(device))
            criterion = nn.CrossEntropyLoss()
            print("Start training process")
            for epoch in range(args.epoch):
                model.train()
                total, epoch_correct, epoch_loss = 0, 0, 0
                for data in trainloader:
                    inputs, labels = data[0].to(device), data[1].to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    epoch_correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                wandb.log({"Training loss per epoch":epoch_loss / total})
                wandb.log({"Training accuracy per epoch":epoch_correct / total * 100})
                # saving check point
                if os.path.isdir("./checkpoint") == False:
                    os.mkdir("checkpoint")
                PATH = "./checkpoint/epoch" + str(epoch + 1)
                torch.save(model.state_dict(), PATH)
                model.eval()
                total, epoch_correct, epoch_loss = 0, 0, 0
                with torch.no_grad():
                    for data in valloader:
                        inputs, labels = data[0].to(device), data[1].to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        epoch_correct += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        epoch_loss += loss.item()
                    wandb.log({"Validation loss per epoch":epoch_loss / total})
                    wandb.log({"Validation accuracy per epoch":epoch_correct / total * 100})

        if args.test:
            print("Using Convolution Neural Network model")
            PATH = "./checkpoint/epoch" + str(args.n)
            model.load_state_dict(torch.load(PATH))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            with torch.no_grad():
                for data in testloader:
                    inputs= data[0].to(device)
                    outputs = model(inputs)
                save(softmax(np.array(outputs.cpu())), "CNN")