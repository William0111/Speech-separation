#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 04:25:09 2019

@author: william
"""

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



def read_Flac(filename):
    data, fs = \
        sf.read('/home/william/Documents/code/Desk top12.10from_disk/flac_dataforcnn/' + str(filename) + '.flac')
    return data, fs

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
    s = np.where(abs(s) > 1, s, 0)
    s = np.where(abs(s) < 1, s, 1)
    s = np.real(s)
    # masked_signal_spectrogram = np.logical_and(spectrogram, s) * spectrogram
    return s  # masked_signal_spectrogram

fftLen = 512
win = hamming(fftLen)
step = int(fftLen / 4)



a1 = read_Flac(10001)

x1 = read_Flac(10001)[0]
x2 = x1[:16000 * 2]


spec = np.zeros(247 * 512)
x1_spec = abs(stft(x2, win, step)).reshape(247 * 512)

x2_spec = x1_spec
MAX_x1_spec=np.max(x2_spec)
x3_spec = x2_spec/MAX_x1_spec
spec = np.vstack((spec, x3_spec))

x1_mask = get_mask(x2_spec).reshape(247 * 512)
x2_mask = x1_mask.reshape(247,512)
x2_spec0 = x2_spec.reshape(247,512)

x3_spec0 = x3_spec.reshape(247,512)
