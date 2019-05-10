#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:05:01 2018

@author: William
"""

import matplotlib.pyplot as plt

import soundfile as sf
import scipy
import numpy as np
from scipy import signal
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator



#loading wav
signalData1, samplingFrequency1  = \
    sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/10005.flac')


print(len(signalData1))
print(signalData1)


f, t, Sxx = signal.spectrogram(signalData1, samplingFrequency1, mode='complex')







