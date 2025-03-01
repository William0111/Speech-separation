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
    s = np.where(abs(s) > 1, s, 0)
    s = np.where(abs(s) < 1, s, 1)
    s = np.real(s)
    # masked_signal_spectrogram = np.logical_and(spectrogram, s) * spectrogram
    return s  # masked_signal_spectrogram


###  Global Parameter part ###
data = read_Flac(10005)[0]
fs = read_Flac(10005)[1]
data = data[:16000 * 2]  # must read data befor this step, so can cut into same length
fftLen = 512
win = hamming(fftLen)
step = int(fftLen / 4)







clean_mask = np.loadtxt("clean_mask.txt")  
spec_train = np.loadtxt("spec_train.txt")  
clean_mask_test = np.loadtxt("clean_mask_test.txt")  
spec_train_test = np.loadtxt("spec_train_test.txt")




for i in range(1000):
    batch_xs = spec_train
    batch_ys = clean_mask

    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    # if i % 5 == 0:
    # a = tf.Print(prediction, [prediction])
    # print (sess.run(prediction))
    # result = a + 0
    if i % 1 == 0:
        print(compute_accuracy(
            spec_train_test, clean_mask_test))

#pre = sess.run(prediction,feed_dict={xs: batch_xs, ys: batch_ys})

# ### Check part
# a = stft(data, win, step)
# b = a.shape


# sess.run(prediction)
