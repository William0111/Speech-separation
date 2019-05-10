#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 04:05:40 2019

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



#a1 = sf.read('/home/william/Documents/code/Desk top12.10from_disk/flac_dataforcnn/' + str(10001) + '.flac')
a2 = sf.read('/home/william/Documents/code/After0303/get1000out_try/' + str(100001) + '.flac')

#aa1 = sf.read('/home/william/Documents/code/Desk top12.10from_disk/flac_dataforcnn/' + str(10001) + '.flac')[0]
aa2 = sf.read('/home/william/Documents/code/After0303/get1000out_try/' + str(100001) + '.flac')[0]