import argparse
import numpy as np
import csv
import pdb
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_data():
    train_dir, test_dir = './fgd_data/train.csv', './fgd_data/test.csv'
    train_x, train_y, test= [], np.zeros((20621, 5)), []
    with open(train_dir) as d:
        rows = csv.reader(d)
        for i, row in enumerate(rows):
            if i > 0:
                row = [float(a) for a in row[2:]]
                train_x.append(row[:-1])
                train_y[i-1][int(row[-1]) - 1] = 1
    with open(test_dir) as d:
        rows = csv.reader(d)
        for i, row in enumerate(rows):
            if i > 0:
                row = [float(a) for a in row[2:]]
                test.append(row)

    return train_x, train_y, test

# there are 20621 training datas, 500 test datas
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Learning Fianl Project')
    parser.add_argument('--method', type=str, default="CNN", metavar='M',
                        help='Input the method you want to use, e.g. LR(Linear Regression), KNN(K-Nearest Neighbor), DT(Decesion Tree), RF(Random Forest) and CNN(Convolutional Neural Network)(default: CNN)')
    parser.add_argument('--neighbor', type=int, default=10, metavar='N',
                        help='Input number of neighbor (default: 10)')
    parser.add_argument('--depth', type=int, default=1, metavar='D',
                        help='Input max depth for Random Forest (default: 1)')
    parser.add_argument('--state', type=int, default=1, metavar='S',
                        help='Input random state for Random Forest (default: 1)')
    # parser.add_argument('--batchsize', type=int, default=100, metavar='N',
    #                     help='Input batch size for training (default: 100)')
    # parser.add_argument('--epoch', type=int, default=200, metavar='N',
    #                     help='Input number of epoch for training (default: 200)')
    # parser.add_argument('--optimizer', type=str, default='Adam', metavar='o',
    #                     help='Input the optimizer, Adam or SGD (default: Adam)')
    # parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
    #                     help='Input the learning rate of the optimizer (default: 0.001)')
    # parser.add_argument('--train', action='store_true', default=False,
    #                     help='Use training mode')
    # parser.add_argument('--evaluate', action='store_true', default=False,
    #                     help='Use validation mode')
    # parser.add_argument('--test', action='store_true', default=False,
    #                     help='Use testing mode')
    train_x, train_y, test = get_data()
    args = parser.parse_args()
    if args.method == "LR":
        print("Using Linear Regression")
        model = LinearRegression().fit(train_x, train_y)
        prediction = model.predict(test)
        print(prediction)

    if args.method == "KNN":
        print("Using K-Nearest Neighbor")
        model = KNeighborsClassifier(n_neighbors=args.neighbor).fit(train_x, train_y)
        prediction = model.predict(test)
        print(prediction)

    if args.method == "DT":
        print("Using Decision Tree")
        model = DecisionTreeClassifier().fit(train_x, train_y)
        prediction = model.predict(test)
        print(prediction)

    if args.method == "RF":
        print("Using Random Forest")
        model = RandomForestClassifier(max_depth=args.depth, random_state=args.state).fit(train_x, train_y)
        prediction = model.predict(test)
        print(prediction)

    if args.method == "CNN":
        print("Using Convolution Neural Network")