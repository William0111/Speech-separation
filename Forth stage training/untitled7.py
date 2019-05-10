#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 06:04:31 2019

@author: william
"""



# The aim is to generate data
# The difficulty might come across will be the input data into train file or test file

import IPython.display
from ipywidgets import interact, interactive, fixed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import copy
from scipy.fftpack import fft
from scipy import ifft
from scipy.signal import butter, lfilter
import scipy.ndimage
import soundfile as sf
from scipy import ceil, complex64, float64, hamming, zeros
import tensorflow as tf


### Function part ###

# read file

def read_Flac(filename):
    data, fs = \
        sf.read('/home/william/Documents/code/After0303/train-clean-100/' + str(filename) + '.flac')
    return data, fs

number = 28000
j = 0
LEN = np.zeros(1)
while j < number:
    x1 = read_Flac(10000 + j)[0]
    x2=len(x1)
    LEN = np.vstack((LEN, x2))
    #LEN.append(str(len(x1)))
    #if LEN<32000:
        #print(j)
    print(j)
    j = j+1
LEN = np.delete(LEN, 0, 0)