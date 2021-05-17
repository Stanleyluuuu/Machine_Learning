'''
NTHU EE Machine Learning HW2
Author: 盧冠維
Student ID: 109061621
'''
import numpy as np
import pandas as pd
import argparse
from scipy import stats

class MaxLikeHood():
    def __init__(self, x):

        pass


    def fit(self, x, y): # fit the model to get the best weight
        x_inv = np.linalg.pinv(x)
        self.weights_leastsquare = np.dot(x_inv, y)
            

    def predict(self, test_x): # predict the result for validation set (or testing set)
        predictions = np.dot(test_x, self.weights_leastsquare)

        return predictions


class Bayesian():
    def __init__(self, features, alpha=0.5, beta=1):
        self.n_features = features
        self.alpha = alpha
        self.beta = beta
        self.mean = np.zeros(features)
        self.cov_inv = np.identity(features) / alpha


    def fit(self, x, y): # fit the model to get the best weight
        for i, ins in enumerate(x):
            cov_inv = self.cov_inv + self.beta * np.outer(ins, ins)
            cov = np.linalg.inv(cov_inv)
            mean = np.dot(cov, (np.dot(self.cov_inv, self.mean) + self.beta * y[i] * ins))
            self.cov_inv = cov_inv
            self.mean = mean


    def predict(self, test_x): # predict the result for validation set (or testing set)
        prediction = []
        for x in test_x:
            y_pred_mean = np.dot(x, self.mean)
            w_cov = np.linalg.inv(self.cov_inv)
            y_pred_var = (1 / self.beta) + x @ w_cov @ x.T
            prediction.append(stats.norm(loc=y_pred_mean, scale=y_pred_var ** .5).mean())

        return np.array(prediction)
            

# do not change the name of this function
def BLR(train_data, test_data_feature, O1=2, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    Phi_train, y = get_phi(train_data, O1, O2, label=True)
    Phi_test = get_phi(test_data_feature, O1, O2)
    model = Bayesian(features=Phi_train.shape[1])
    model.fit(Phi_train, y)
    y_BLRprediction = model.predict(Phi_test)

    return y_BLRprediction


# do not change the name of this function
def MLR(train_data, test_data_feature, O1=2, O2=2):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    Phi_train, y = get_phi(train_data, O1, O2, label=True)
    Phi_test = get_phi(test_data_feature, O1, O2)
    model = MaxLikeHood(Phi_train)
    model.fit(Phi_train, y)
    y_MLLSprediction = model.predict(Phi_test)

    return y_MLLSprediction


def get_phi(data, O1, O2, label=False): # function use to convert raw data into a feature vector
    x1_max, x2_max, x1_min, x2_min = data.max(axis=0)[0], data.max(axis=0)[1], data.min(axis=0)[0], data.min(axis=0)[1] # get the max and min of 2 features(GRE and TOEFL score)
    s1, s2 = ((x1_max - x1_min) / (O1 - 1)), ((x2_max - x2_min) / (O2 - 1)) # calculating s1 and s2
    Phi = []
    for i in range(1, O1+1):
        for j in range(1, O2+1):
            mui, muj = (s1 * (i - 1) + x1_min), (s2 * (j - 1) + x2_min) # calculating mu_i and mu_j
            px = np.exp(-(((data[:, 0] - mui) ** 2) / (2 * s1 ** 2)) - (((data[:, 1] - muj) ** 2) / (2 * s2 ** 2))) # gaussian basis function
            px = px - px.mean(axis=0)
            Phi.append(px)
    Phi.append(data[:, 2]) # adding research experience 1 or 0
    Phi.append(np.ones(len(data)))
    if label: # if need label, generate from input data and return
        y = data[:, 3]

        return np.array(Phi).T, y
    else: # if not, return Phi only
        
        return np.array(Phi).T


def CalMSE(data, prediction): # Mean Square Error
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

    predict_BLR = BLR(data_train, data_test_feature, O1=2, O2=5)
    predict_MLR = MLR(data_train, data_test_feature, O1=2, O2=2)

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()