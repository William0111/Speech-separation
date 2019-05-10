#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 01:56:26 2018

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

# download data
X_train = spec_train = np.loadtxt("clean_spec_train_1000.txt")

y_train = clean_mask = np.loadtxt("clean_mask_train_1000.txt")

print(X_train)
print(y_train)
