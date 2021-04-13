import numpy as np
import csv
import pdb
import random

###################################################################################################
# Read in the instances from csv file, then cluster them into three lists according to the label. #
###################################################################################################
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
train_set = []
test_set = []
for data in datas:
    random.shuffle(data)
    for i, d in enumerate(data):
        if i < 18:
            test_set.append(d)
        else:
            train_set.append(d)
#######################################################
# Save the testing set and training set as csv files. #
#######################################################
with open('test.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(test_set)

with open('train.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(train_set)
##################################################
# Load in the datas from train.csv and test.csv. #
##################################################
training_data = []
testing_data = []
with open('train.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        training_data.append(row)
with open('test.csv') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        testing_data.append(row)

# pdb.set_trace()