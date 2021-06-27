import argparse
import numpy as np
import pdb
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from final_utils import *
import wandb

            
# there are 20621 training datas, 500 test datas
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Learning Fianl Project')
    parser.add_argument('--method', type=str, default="CNN", metavar='M',
                        help='Input the method you want to use, e.g. LR(Linear Regression), KNN(K-Nearest Neighbor), DT(Decesion Tree), RF(Random Forest) and CNN(Convolutional Neural Network)(default: CNN)')
    parser.add_argument('--neighbor', type=int, default=3, metavar='N',
                        help='Input number of neighbor (default: 3)')
    parser.add_argument('--depth', type=int, default=20, metavar='D',
                        help='Input max depth for Random Forest (default: 20)')
    parser.add_argument('--state', type=int, default=15, metavar='S',
                        help='Input random state for Random Forest (default: 15)')
    parser.add_argument('--batchsize', type=int, default=200, metavar='N',
                        help='Input batch size for training (default: 100)')
    parser.add_argument('--epoch', type=int, default=500, metavar='N',
                        help='Input number of epoch for training (default: 500)')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='o',
                        help='Input the optimizer, Adam or SGD (default: Adam)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Input the learning rate of the optimizer (default: 0.001)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Use training mode')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Use testing mode')
    parser.add_argument('--find', action='store_true', default=False,
                        help='Use finding mode')
    parser.add_argument('--n', type=int, default=500,
                        help='For testing')        
    args = parser.parse_args()
    
    # split()
    if args.method == "LR":
        # train_x, train_y, test = get_data(args)
        train_x, train_y, val_x, val_y, test = get_data(args)
        print("Using Linear Regression")
        model = LinearRegression().fit(train_x, train_y)
        p_train = model.predict(train_x)
        soft_pred = softmax(p_train)
        acc = sum(np.argmax(soft_pred, axis=1) == np.argmax(train_y, axis=1)) / len(train_x)
        loss = get_loss(soft_pred, train_y)
        print("Accuracy on training set = {e1:.3f}%, loss = {e2:.3f}".format(e1=acc, e2=loss))

        p_val = model.predict(val_x)
        soft_pred_v = softmax(p_val)
        acc_v = sum(np.argmax(soft_pred_v, axis=1) == np.argmax(val_y, axis=1)) / len(val_x)
        loss_v = get_loss(soft_pred_v, val_y)
        print("Accuracy on validation set = {e1:.3f}%, loss = {e2:.3f}".format(e1=acc_v, e2=loss_v))

        # prediction = model.predict(test)
        # soft_pp = softmax(prediction)
        # save(soft_pp, "LR")

    if args.method == "KNN":
        # train_x, train_y, test = get_data(args)
        train_x, train_y, val_x, val_y, test = get_data(args)
        print("Using K-Nearest Neighbor")
        model = KNeighborsClassifier(n_neighbors=args.neighbor).fit(train_x, train_y)
        p_train = model.predict(train_x)
        acc = sum(np.argmax(p_train, axis=1) == np.argmax(train_y, axis=1)) / len(train_x)
        loss = get_loss(p_train, train_y, epsilon=1e-10)
        print("Accuracy on training set = {e1:.3f}%, loss = {e2:.3f}".format(e1=acc, e2=loss))

        p_val = model.predict(val_x)
        acc_v = sum(np.argmax(p_val, axis=1) == np.argmax(val_y, axis=1)) / len(val_x)
        loss_v = get_loss(p_val, val_y, epsilon=1e-10)
        print("Accuracy on validation set = {e1:.3f}%, loss = {e2:.3f}".format(e1=acc_v, e2=loss_v))

        # prediction = model.predict(test)
        # save(prediction, "KNN")

    if args.method == "DT":
        # train_x, train_y, test = get_data(args)
        train_x, train_y, val_x, val_y, test = get_data(args)
        print("Using Decision Tree")
        model = DecisionTreeClassifier().fit(train_x, train_y)
        p_train = model.predict(train_x)
        acc = sum(np.argmax(p_train, axis=1) == np.argmax(train_y, axis=1)) / len(train_x)
        loss = get_loss(p_train, train_y, epsilon=1e-10)
        print("Accuracy on training set = {e1:.3f}%, loss = {e2:.3f}".format(e1=acc, e2=loss))

        p_val = model.predict(val_x)
        acc_v = sum(np.argmax(p_val, axis=1) == np.argmax(val_y, axis=1)) / len(val_x)
        loss_v = get_loss(p_val, val_y, epsilon=1e-10)
        print("Accuracy on validation set = {e1:.3f}%, loss = {e2:.3f}".format(e1=acc_v, e2=loss_v))

        prediction = model.predict(test)
        save(prediction, "DT")

    if args.method == "RF":
        # train_x, train_y, test = get_data(args)
        train_x, train_y, val_x, val_y, test = get_data(args)
        print("Using Random Forest")
        model = RandomForestClassifier(max_depth=args.depth, random_state=args.state).fit(train_x, train_y)
        p_train = model.predict(train_x)
        acc = sum(np.argmax(p_train, axis=1) == np.argmax(train_y, axis=1)) / len(train_x)
        loss = get_loss(p_train, train_y, epsilon=1e-10)
        print("Accuracy on training set = {e1:.3f}%, loss = {e2:.3f}".format(e1=acc, e2=loss))

        p_val = model.predict(val_x)
        acc_v = sum(np.argmax(p_val, axis=1) == np.argmax(val_y, axis=1)) / len(val_x)
        loss_v = get_loss(p_val, val_y, epsilon=1e-10)
        print("Accuracy on validation set = {e1:.3f}%, loss = {e2:.3f}".format(e1=acc_v, e2=loss_v))

        prediction = model.predict(test)
        save(prediction, "RF")

    if args.method == "CNN":
        trainloader, valloader, testloader = get_data(args, row=1)
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
            model = CNN()
            if args.optimizer == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            elif args.optimizer == "Adam":
                optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
            w = torch.tensor((20621 / 7064, 20621 / 8195, 20621 / 3855, 20621 / 1031, 20621 / 476))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            criterion = nn.CrossEntropyLoss(weight=w.to(device))
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
                PATH = "./checkpoint/epoch" + str(epoch + 1)
                torch.save(model.state_dict(), PATH)

        if args.find:
            # print("Using Convolution Neural Network model")
            model = CNN()
            result = []
            w = torch.tensor((20621 / 7064, 20621 / 8195, 20621 / 3855, 20621 / 1031, 20621 / 476))
            for epoch in range(args.epoch):
                PATH = "./checkpoint/epoch" + str(epoch + 1)
                model.load_state_dict(torch.load(PATH))
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)
                criterion = nn.CrossEntropyLoss(w.to(device))
                model.eval()
                with torch.no_grad():
                    total, epoch_correct, epoch_loss = 0, 0, 0
                    # for data in trainloader:
                    for data in valloader:
                        inputs, labels = data[0].to(device), data[1].to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        epoch_correct += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        epoch_loss += loss.item()
                result.append((epoch+1, epoch_loss / total, epoch_correct / total * 100))
            a = min(result, key=lambda x: x[1])
            print("The lowest loss =", a)
            b = max(result, key=lambda x: x[2])
            print("The highest accuracy =", b)
        if args.test:
            print("Using Convolution Neural Network model")
            model = CNN()
            PATH = "./checkpoint/epoch" + str(args.n)
            model.load_state_dict(torch.load(PATH))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            with torch.no_grad():
                for data in testloader:
                    inputs= data[0].to(device)
                    outputs = model(inputs)
                save(softmax(np.array(outputs.cpu())))