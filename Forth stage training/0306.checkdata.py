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

y_test = np.loadtxt("0306_spec_train_20000data_3columns.txt")
y_test = y_test[0:1000]
print(X_train)
print(y_train)


y_mask = y_test.reshape(1000,1536)
plt.imshow(abs(y_mask[:, : int(512 / 2 + 1)].T), aspect = "auto", cmap=plt.cm.afmhot, origin = "lower")
plt.title("Lable_Mask", fontsize = 20)
plt.show()
