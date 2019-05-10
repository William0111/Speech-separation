#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 23:09:40 2019

@author: william
"""

#%matplotlib inline
import matplotlib


import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, Flatten,LSTM, TimeDistributed, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt

TIME_STEPS = 10
INPUT_SIZE = 512
BATCH_SIZE = 250
BATCH_INDEX = 0
OUTPUT_SIZE = 5120
#CELL_SIZE = 800
LR = 0.001


           
           
# loading data
X_train = np.loadtxt("0131_spec_train_1000*5120.txt")
y_train = np.loadtxt("0131_mask_train_1000*5120.txt")

X_test = np.loadtxt("0131_spec_test_1000*5120.txt")
y_test = np.loadtxt("0131_mask_test_1000*5120.txt")

X_train_reshaped = X_train.reshape(-1, 10, 512)

y_train_reshaped = y_train.reshape(-1, 10, 512)

t = []
for i in range(1000):
    y
    t.append(i%10)
print(t)
    













# batch_size=y_test.shape[0]
'''
# data pre-processing
X_train = X_train.reshape(-1, 10, 512)
#y_train = y_train.reshape(-1, 3, 512)
X_test = X_test.reshape(-1, 10, 512)
#y_test = y_test.reshape(-1, 3, 512)
'''










