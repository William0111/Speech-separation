#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:35:40 2019

@author: william
"""



# import part
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt

# loading data
X_train = np.loadtxt("0126_spec_train_1000.txt")
y_train = np.loadtxt("0126_mask_train_1000.txt")

X_test = np.loadtxt("0126_spec_test_1000.txt")
y_test = np.loadtxt("0126_mask_test_1000.txt")

X_train = X_train.reshape(-1,1,3,512)
y_train = y_train.reshape(-1,1,3,512)
X_test = X_test.reshape(-1,1,3,512)
y_test = y_test.reshape(-1,1,3,512)