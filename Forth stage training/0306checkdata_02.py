

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
        sf.read('/home/william/Documents/code/Desk top12.10from_disk/flac_dataforcnn/' + str(filename) + '.flac')
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
fftLen = 512
win = hamming(fftLen)
step = int(fftLen / 4)


# Generate training data
def generate_spectrogram_data(start_from=100001, number=1000, noise_power=0, length=3):
    spec = np.zeros(512*length)
    j = 0
    while j < number:
        x1 = read_Flac(start_from + j)[0]
        x1 = x1[:16000 * 2]

        noise_with_power = np.var(x1) * noise_power
        noise = np.random.normal(scale=noise_with_power, size=x1.shape)
        x1 = x1 + noise

        x1_spec = abs(stft(x1, win, step)).reshape(247 * 512)
        high_boundary = 247 * 512
        low_boundary = (247 - length) * 512
        x1_spec = x1_spec[low_boundary:high_boundary]  # cut between low boundry and high boundry
        x1_spec = x1_spec / np.max(x1_spec)
        spec = np.vstack((spec, x1_spec))

        j = j + 1
    spec = np.delete(spec, 0, 0)
    return spec


def generate_mask_data(start_from=100001, number=1000, noise_power=0, length=3):
    mask = np.zeros(512*length)
    j = 0
    while j < number:
        x1 = read_Flac(start_from + j)[0]
        x1 = x1[:16000 * 2]

        noise_with_power = np.var(x1) * noise_power
        noise = np.random.normal(scale=noise_with_power, size=x1.shape)
        x1 = x1 + noise

        x1_spec = abs(stft(x1, win, step))
        x1_mask = get_mask(x1_spec).reshape(247 * 512)
        high_boundary = 247 * 512
        low_boundary = (247 - length) * 512
        x1_mask = x1_mask[low_boundary:high_boundary]  # cut between low boundry and high boundry
        x1_mask = x1_mask
        mask = np.vstack((mask, x1_mask))

        # x2_spec = abs(stft(x2, win, step).reshape(247 * 512))
        #
        # x2_spec = x2_spec[247 * 490:247 * 500]
        # x2_spec = x2_spec / np.max(x2_spec)
        # spec_train = np.vstack((spec_train, x2_spec))

        j = j + 1
    mask = np.delete(mask, 0, 0)
    return mask

#clear_spec_train_1000 = generate_spectrogram_data(start_from=100001, number=1000, noise_power=0, length=3)
#clear_mask_train_1000 = generate_mask_data(start_from=100001, number=1000, noise_power=0, length=3)

#np.savetxt('0306_spec_train_1000data_3columns.txt', clear_spec_train_1000, delimiter = ' ')
#np.savetxt('0306_mask_train_1000data_3columns.txt', clear_mask_train_1000, delimiter = ' ')

#try to plot to check
x1 = read_Flac(10000 + 1)[0]
x1 = x1[:16000 * 3]


x1_spec = abs(stft(x1, win, step)).reshape(372 * 512)

x1_mask = get_mask(x1_spec).reshape(372 * 512)


#high_boundary = 247 * 512
#low_boundary = (247 - 200) * 512
#x1_spec = x1_spec[low_boundary:high_boundary]  # cut between low boundry and high boundry
x1_spec = x1_spec / np.max(x1_spec)



y_mask = x1_spec.reshape(372,512)
plt.imshow(abs(y_mask[:, : int(512 / 2 + 1)].T), aspect = "auto", cmap=plt.cm.afmhot, origin = "lower")
plt.title("Lable_Mask", fontsize = 20)
plt.show()











