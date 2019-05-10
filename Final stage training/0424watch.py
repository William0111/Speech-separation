#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:00:48 2019

@author: william
"""
# import part
import numpy as np

import matplotlib.pyplot as plt

# loading data
X_train = np.loadtxt("1209_mask_test_1000.txt")
y_train = np.loadtxt("1209_spec_test_1000.txt")

X_train2 = np.loadtxt("clean_spec_test_1000.txt")
y_train2 = np.loadtxt("clean_mask_test_1000.txt")

y_mask = y_train2.reshape(1000,2470)
plt.imshow(abs(y_mask[:, : int(512 / 2 + 1)].T), aspect = "auto", cmap=plt.cm.afmhot, origin = "lower")
plt.title("Lable_Mask", fontsize = 20)
plt.show()