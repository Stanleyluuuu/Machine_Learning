import numpy as np
import csv
import random
from sklearn.decomposition import PCA
import pdb
import matplotlib.pyplot as plt

class HW1:
    def __init__(self):
        pass

    def save_csv(self, filename, set): # save the datas as a csv file
        with open(filename, 'w') as f:
            write = csv.writer(f)
            write.writerows(set)

    def load_csv(self, filename): # load the datas from the csv file that I just saved, and devide into training and testing set
        datas, x, y = [], [], []
        with open(filename) as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                datas.append(row)
        for data in datas:
            data = [float(item) for item in data] # turn the elements from string to float
            x.append(data[1:])
            y.append(int(data[:1][0]))
        
        return np.array(x), np.array(y)

    def get_mvp(self, x, y): # get mean, variance for each class and each feature, and get the prior probability for three classes
        # devide in to three classes
        type_1, type_2, type_3 = [], [], []
        for i, classes in enumerate(y):
            if classes == 1:
                type_1.append(x[i])
            elif classes == 2:
                type_2.append(x[i])
            else:
                type_3.append(x[i])
        # get mean and variance for 3 classes and 13 features
        t1_mean, t1_var = np.array(type_1).mean(axis=0), np.array(type_1).var(axis=0)
        t2_mean, t2_var = np.array(type_2).mean(axis=0), np.array(type_2).var(axis=0)
        t3_mean, t3_var = np.array(type_3).mean(axis=0), np.array(type_3).var(axis=0)
        self.mean, self.var = np.stack((t1_mean, t2_mean, t3_mean), axis=0), np.stack((t1_var, t2_var, t3_var), axis=0)
        # get prior probability
        self.prior = (len(type_1) / self.total, len(type_2) / self.total, len(type_3) / self.total)

        return self.mean, self.var, self.prior

    def likelihood(self, class_idx, x): # get the likelihood function
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        likelihoood = numerator / denominator

        return likelihoood

    def posterior(self, x): # calcutate the map a posteriori probability
        posteriors = []
        for i in range(self.count): # calculate the log likelihood function and log prior probability, then add them together.
            prior = np.log(self.prior[i]) 
            likelihood = np.sum(np.log(self.likelihood(i, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def fit(self, x, y): # fit the parameters to the model
        self.classes = np.unique(y)
        self.count = len(self.classes)
        self.feature_nums = x.shape[1]
        self.total = x.shape[0]
        self.get_mvp(x, y)

    def predict(self, x_test): # evaluate the predition using test set
        predictions = [self.posterior(f) for f in x_test]

        return predictions
    
    def accuracy(self, y_test, y_pred): # calculate the accuracy of the prediction
        accuracy = np.sum(y_test == y_pred) / len(y_test)

        return accuracy

    def visualization(self, x, y):
        type_1, type_2, type_3 = [], [], []
        for i, classes in enumerate(y):
            if classes == 1:
                type_1.append(x[i])
            elif classes == 2:
                type_2.append(x[i])
            else:
                type_3.append(x[i])
        pca = PCA(n_components=2)

        return pca.fit_transform(np.array(type_1)), pca.fit_transform(np.array(type_2)), pca.fit_transform(np.array(type_3))

if __name__ == "__main__":
    ###################################################################################################
    # Read in the instances from csv file, then cluster them into three lists according to the label. #
    ###################################################################################################
    hw1 = HW1()
    type_1 = []
    type_2 = []
    type_3 = []
    with open('Wine.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if row[0] == '1':
                type_1.append(row)
            elif row[0] == '2':
                type_2.append(row)
            else:
                type_3.append(row)
    ###################################################################################################
    # Random sample 18 instances from three different classes as testing set, others as training set. #
    ###################################################################################################
    datas = (type_1, type_2, type_3)
    train_set, test_set = [], []
    for data in datas:
        random.shuffle(data)
        for i, d in enumerate(data):
            if i < 18:
                test_set.append(d)
            else:
                train_set.append(d)
    #########################################################################################
    # Save the testing set and training set as csv files, and load the datas from csv files #
    #########################################################################################
    hw1.save_csv('test.csv', test_set)
    hw1.save_csv('train.csv', train_set)
    training_x, training_y = hw1.load_csv('train.csv')
    testing_x, testing_y = hw1.load_csv('test.csv')
    ###################################################################################################
    # Fit the datas to the model, then calculate the accuracy and how many incorrect labels are there #
    ###################################################################################################
    hw1.fit(training_x, training_y)
    y_pred = hw1.predict(testing_x)
    accuracy = hw1.accuracy(testing_y, y_pred)
    print("Mislabeled %d points out of a total %d points, accuracy is %.3f " % ((testing_y != y_pred).sum(), testing_x.shape[0], accuracy))
    ############################################################################
    # Use pca to reduce the dimension to 2 then visualize test set as a figure #
    ############################################################################
    pca1, pca2, pca3 = hw1.visualization(testing_x, testing_y)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(pca1[:, 0], pca1[:, 1], color='r', label='Type 1')
    ax1.scatter(pca2[:, 0], pca2[:, 1], color='g', label='Type 2')
    ax1.scatter(pca3[:, 0], pca3[:, 1], color='b', label='Type 3')
    ax1.legend()
    plt.savefig("result.png")
    plt.show()