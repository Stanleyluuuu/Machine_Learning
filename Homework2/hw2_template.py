'''
NTHU EE Machine Learning HW2
Author: 盧冠維
Student ID: 109061621
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse
import pdb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# do not change the name of this function
def BLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    x1_max, x2_max, x1_min, x2_min = train_data.max(axis=0)[0], train_data.max(axis=0)[1], train_data.min(axis=0)[0], train_data.min(axis=0)[1] # get the max and min of 2 features(GRE and TOEFL score)
    s1, s2 = ((x1_max - x1_min) / (O1 - 1)), ((x2_max - x2_min) / (O2 - 1)) # calculating s1 and s2
    Phi = []
    for i in range(1, O1+1):
        for j in range(1, O2+1):
            k = O2 * (i - 1) + j
            mui, muj = (s1 * (i - 1) + x1_min), (s2 * (j - 1) + x2_min) # calculating mu_i and mu_j
            a = np.exp(-(((train_data[:, 0] - mui) ** 2) / (2 * s1 ** 2)) - (((train_data[:, 1] - muj) ** 2) / (2 * s2 ** 2))) # gaussian basis function
            Phi.append(a)
    Phi.append(train_data[:, 2]) # adding research experience 1 or 0
    Phi.append(np.ones(300)) # adding bias
    ##### We have convert the input into a feature vector constructed by gaussian basis function, research experience and bias #####
    pdb.set_trace()

    return y_BLRprediction 


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    x1_max, x2_max, x1_min, x2_min = train_data.max(axis=0)[0], train_data.max(axis=0)[1], train_data.min(axis=0)[0], train_data.min(axis=0)[1] # get the max and min of 2 features(GRE and TOEFL score)
    s1, s2 = ((x1_max - x1_min) / (O1 - 1)), ((x2_max - x2_min) / (O2 - 1)) # calculating s1 and s2
    Phi = []
    for i in range(1, O1+1):
        for j in range(1, O2+1):
            k = O2 * (i - 1) + j
            mui, muj = (s1 * (i - 1) + x1_min), (s2 * (j - 1) + x2_min) # calculating mu_i and mu_j
            a = np.exp(-(((train_data[:, 0] - mui) ** 2) / (2 * s1 ** 2)) - (((train_data[:, 1] - muj) ** 2) / (2 * s2 ** 2))) # gaussian basis function
            Phi.append(a)
    Phi.append(train_data[:, 2]) # adding research experience 1 or 0
    Phi.append(np.ones(300)) # adding bias
    ##### We have convert the input into a feature vector constructed by gaussian basis function, research experience and bias #####
    pdb.set_trace()

    return y_MLLSprediction 


def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=5)
    parser.add_argument('-O2', '--O_2', type=int, default=5)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))

if __name__ == '__main__':
    main()
