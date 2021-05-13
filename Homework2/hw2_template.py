'''
NTHU EE Machine Learning HW2
Author: 盧冠維
Student ID: 109061621
'''
import numpy as np
import pandas as pd
import math
from scipy.stats import multivariate_normal as mv_norm
import argparse
import pdb
from sklearn import linear_model

# do not change the name of this function
def BLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    Phi_train, y = get_phi(train_data, O1, O2, label=True)
    Phi_test = get_phi(test_data_feature, O1, O2)
    regr = linear_model.LinearRegression()
    # regr = linear_model.BayesianRidge()
    regr.fit(Phi_train, y)
    y_BLRprediction = regr.predict(Phi_test)
    # pdb.set_trace()

    return y_BLRprediction 


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    Phi_train, y = get_phi(train_data, O1, O2, label=True)
    Phi_test = get_phi(test_data_feature, O1, O2)
    pdb.set_trace()

    return y_MLLSprediction 


def get_phi(data, O1, O2, label=False): # function use to convert raw data into a feature vector
    x1_max, x2_max, x1_min, x2_min = data.max(axis=0)[0], data.max(axis=0)[1], data.min(axis=0)[0], data.min(axis=0)[1] # get the max and min of 2 features(GRE and TOEFL score)
    s1, s2 = ((x1_max - x1_min) / (O1 - 1)), ((x2_max - x2_min) / (O2 - 1)) # calculating s1 and s2
    Phi = []
    for i in range(1, O1+1):
        for j in range(1, O2+1):
            mui, muj = (s1 * (i - 1) + x1_min), (s2 * (j - 1) + x2_min) # calculating mu_i and mu_j
            px = np.exp(-(((data[:, 0] - mui) ** 2) / (2 * s1 ** 2)) - (((data[:, 1] - muj) ** 2) / (2 * s2 ** 2))) # gaussian basis function
            Phi.append(px)
    Phi.append(data[:, 2]) # adding research experience 1 or 0
    Phi.append(np.ones(len(data)))
    if label: # if need label, generate from input data and return
        y = data[:, 3]

        return np.array(Phi).T, y
    else: # if not, return Phi only
        
        return np.array(Phi).T


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
    # predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    # predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    # print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))
    ###############################################################################################################################################################
    results = []
    for i in range(2, 6):
        for j in range(2, 6):
            O_1 = i
            O_2 = j
            predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
            results.append((O_1, O_2, CalMSE(predict_BLR, data_test_label)))
            print('While O1 = {e2}, O2 = {e3}, MSE of BLR = {e1}'.format(e1=CalMSE(predict_BLR, data_test_label), e2=i, e3=j))
    a = min(results, key=lambda x: x[2])
    print(a)

if __name__ == '__main__':
    main()
