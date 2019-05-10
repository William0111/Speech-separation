# This is a summary of everything I have done before

# irst step: import everything we need
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


### Function part ###

# read file
def read_Flac(filename):
    data, fs = \
        sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/' + str(filename) + '.flac')
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


def spectrogram_Original(x):
    spectrogram = stft(x, win, step)
    return spectrogram[:, : int(fftLen / 2 + 1)].T


def spectrogram_Real(spectrogram):
    return np.real(spectrogram)


def spectrogram_Imag(spectrogram):
    return np.imag(spectrogram)


def pretty1_Spectrogram(x, log=True, thresh=5):  # , fft_size=fftLen, step_size=step):
    specgram = np.abs(stft(x, win, step))
    if log == True:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)  # take log
        specgram[specgram < -1 * thresh] = -thresh  # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold

    return specgram


def pretty2_Spectrogram(x, fs):
    Sxx, f, t, im = plt.specgram(data, Fs=int(fs))
    # plt.ylim(0,15000)
    # plt.show()
    not plt.show()
    return Sxx


def plot_signal(x):
    fig, ax = plt.subplots()
    plt.plot(x)
    plt.xlim([0, len(x)])
    s = len(x) / 16000
    ax.set_xticks([0.25 * s, 0.5 * s, 0.75 * s, s])
    plt.title("Signal", fontsize=20)
    plt.show()


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 3))
    cax = ax.matshow(np.transpose(spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                     origin='lower')
    fig.colorbar(cax)
    plt.title('Spectrogram')
    plt.show()
    print('The shape of spectrogram is', spectrogram.shape)
    return spectrogram.shape


def noise_Simple():
    noise_power = np.var(data)
    noise = np.random.normal(scale=noise_power, size=data.shape)
    return noise


def mixed_Signal():
    mix = data + noise_Simple()
    return mix


def get_Mask1(height=-4):
    s = pretty11_Spectrogram(data)
    m1 = np.where(s > height, s, -9)
    m2 = np.where(m1 != -9, m1, 0)
    m3 = np.where(m2 == 0, m2, 1)
    return m2

def istft(spectrogram, win, step):
    M, N = spectrogram.shape
    assert (len(win) == N), "FFT length and window length are different."

    l = (M - 1) * step + N
    x = zeros(l, dtype=float64)
    wsum = zeros(l, dtype=float64)
    for m in range(M):
        start = step * m
        ### Smooth connection
        x[start: start + N] = x[start: start + N] + ifft(spectrogram[m, :]).real * win
        wsum[start: start + N] += win ** 2
    pos = (wsum != 0)
    x_pre = x.copy()
    ### Scaling windows
    x[pos] /= wsum[pos]
    return x

### Parameters part ###
data, fs = read_Flac(10005)
data = data[:16000 * 3]
fftLen = 512
win = hamming(fftLen)
step = int(fftLen / 4)

### Check function and look matrix
# a1 = mixed_Signal()
# a2 = spectrogram_Original(data)
# a3 = pretty1_Spectrogram(data)
# a7 = pretty2_Spectrogram(data, fs)
# a4 = np.abs(a7).T  # abs means abs = np.sqrt(np.square(a_complex)+np.square(a_real))
# a5 = a4 / a4.max()
# a6 = np.log10(a5)
# a_complex = spectrogram_Imag(data)
# a_real = spectrogram_Real(data)
# a_check = a2 - (a_complex * np.complex(0, 1) + a_real)  # Zero!! verified your guess
# abb = np.sqrt(np.square(a_complex) + np.square(a_real))
# a_check2 = a4.T - abb  # Zero!! verified your guess
#
# a8 = np.real(a2) - a7


def pretty11_Spectrogram(x):  # , log=True, thresh=5):  # , fft_size=fftLen, step_size=step):

    specgram = np.abs(pretty2_Spectrogram(x, fs)).T
    if 1 == True:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)  # take log
        specgram[specgram < -9] = -9  # set anything less than the threshold as the threshold
    # else:
    #     specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold

    return specgram


# if 1 == 1:
#     a6[a6 < -9] = -9
# a10 = a6
#
# a9 = get_Mask1()
# print("#########")
# plot_spectrogram(a10)
# plot_spectrogram(a3)
#
# print("Mask")
# plot_spectrogram(get_Mask1())
aa1 = stft(data,win,step)
#aa1 = aa1/np.average(np.abs(aa1))
aa2 = istft(aa1, win, step)

aa3 = np.log10(aa1)

plt.imshow(abs(aa1[:, : int(fftLen / 2 + 1)].T), aspect = "auto", cmap=plt.cm.afmhot, origin = "lower")
plt.title("Spectrogram", fontsize = 20)
plt.show()

# def get_Mask0(height=-4):
#     s = stft(data)
#     m1 = np.where(abs(s) > height, s, -9)
#     m2 = np.where(m1 != -9, m1, 0)
#     m3 = np.where(m2 == 0, m2, 1)
#     return m2
#

s = aa1
s = np.where(abs(s) > 0.1, s, 0)
s = np.where(abs(s) < 0.1, s, 1)
#m1 = np.where(s == 1, s, 0)
s = np.real(s)

plt.imshow(abs(s[:, : int(fftLen / 2 + 1)].T), aspect = "auto", cmap=plt.cm.afmhot, origin = "lower")
plt.title("Spectrogram", fontsize = 20)
plt.show()

masked_aa1 = np.logical_and(aa1,s)*aa1
plt.imshow(abs(masked_aa1[:, : int(fftLen / 2 + 1)].T), aspect = "auto", cmap=plt.cm.afmhot, origin = "lower")
plt.title("Spectrogram", fontsize = 20)
plt.show()








