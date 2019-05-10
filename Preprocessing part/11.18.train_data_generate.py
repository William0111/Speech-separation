#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 03:24:44 2018

@author: William
"""

# The aim is to use DNN or RNN to learn denoising
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
        sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/' + str(filename) + '.flac')
    return data, fs


# get spectrogram
def stft(x, win, step):
    l = len(x)  # length of data
    N = len(win)  # length of window
    M = int(ceil(float(l - N + step) / step))  # Number of Windows in the spectrogram

    new_x = zeros(N + ((M - 1) * step), dtype=float64)
    new_x[: l] = x

    X = zeros([M, N], dtype=complex64)  # Initialization of spectrogram (complex type)
    for m in range(M):
        start = step * m
        X[m, :] = fft(new_x[start: start + N] * win)
    return X

def get_mask(spectrogram):
    s = spectrogram
    s = np.where(abs(s) > 0.5, s, 0)
    s = np.where(abs(s) < 0.5, s, 1)
    s = np.real(s)
    #masked_signal_spectrogram = np.logical_and(spectrogram, s) * spectrogram
    return s #masked_signal_spectrogram


###  Global Parameter part ###
fftLen = 512
win = hamming(fftLen)
step = int(fftLen / 4)


for j in range(100):
    x1 = read_Flac(10001 + j)[0]
    x1 = x1[:16000 * 2]
    noise_power = np.var(x1) * 5
    noise = np.random.normal(scale=noise_power, size=x1.shape)
    x2 = x1 + noise
    mask_clean = np.array([])
    x11 = get_mask(stft(x1, win, step)).reshape(247*512)
    mask_clean = np.append(mask_clean,x11)
    mask_with_noice = []
    mask_with_noice.append(get_mask(stft(x2, win, step)).reshape(247*512))
'''
for jj in range(300):
    x1 = read_Flac(12001 + jj)[0]
    x1 = x1[:16000 * 2]
    noise_power = np.var(x1) * 5
    noise = np.random.normal(scale=noise_power, size=x1.shape)
    x2 = x1 + noise
    t_mask_clean = []
    x1mask = get_mask(stft(x1, win, step)).reshape(247*512)
    t_mask_clean.extend(x1mask)
    t_mask_with_noice = []
    t_mask_with_noice.extend(get_mask(stft(x2, win, step)))

import pandas as pd


### Generate training data ###
j = 0
while j < 100:
    x1 = read_Flac(10101 + j)[0]
    x1 = x1[:16000 * 2]
    
    noise_power = np.var(x1) * 5
    noise = np.random.normal(scale=noise_power, size=x1.shape)
    #x1 = x1 + noise
    
    x1_spec = abs(stft(x1,win,step).reshape(247*512).T)
    x1_spec = x1_spec[247*499:247*510]
    x1_spec = x1_spec/max(x1_spec)
    x1_spec = pd.DataFrame(x1_spec)
    file_path = r'./' + 'test clean spec' + str(j) + '.csv'

    x1_spec.to_csv(file_path,index=False,sep = ',')
    
    
    j = j+1  

    noise_power = np.var(x1) * 5
    noise = np.random.normal(scale=noise_power, size=x1.shape)
    #x2 = x1 + noise
    
    x1mask = get_mask(stft(x1, win, step)).reshape(247*512).T
    x1mask = pd.DataFrame(x1mask).T


    #x2mask = get_mask(stft(x2, win, step)).reshape(247*512).T
    #x2mask = pd.DataFrame(x2mask).T

    t_mask_clean = pd.DataFrame()
    t_mask_clean = t_mask_clean.append(x1mask)
    t_mask_clean = t_mask_clean.append(pd.DataFrame(get_mask(stft(x1[:16000 * 2] + noise, win, step)).reshape(247*512).T).T)
    print('###\nfinished 1 column')
    j = j+1      
    x1spec = pd.DataFrame(stft(x1, win, step))
    x1spec_1 = abs(x1spec[29][20])
'''









