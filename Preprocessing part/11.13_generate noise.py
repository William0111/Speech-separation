"""
This one is basically figure out suitable noise
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import soundfile as sf

data, rate = \
    sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/10001.flac')
data = data[:16000 * 3]

# try to generate noise
# can add into signal directly
# one thing is important, same size, d.type, and length so can add together

noise_power = np.var(data) * 2
noise = np.random.normal(scale=noise_power, size=data.shape)

# define a function to mix
# def mix_with_noise(data, noise):
#     mix = data + noise
#     return mix

mixed_data = data + noise

print('Have a look of data', data)
print('Have a look of noise', noise)
print('Have a look of mixed data', mixed_data)

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 9,
         }

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 14,
         }

plt.subplots(figsize=[10, 10])

ax1 = plt.subplot(4, 1, 1)
ax1.yaxis.set_major_locator(MultipleLocator(1))  # 设定y轴刻度间距

x = range(0, len(data))
plt.plot(x, data, color='black', label='$Data$', linewidth=0.1)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
hl = plt.legend(loc='upper right', prop=font1, frameon=False)  # 绘制图例，指定图例位置

x = range(0, len(mixed_data))
plt.plot(x, mixed_data, color='red', label='$Mixed data$', linewidth=0.1)
plt.legend(loc='upper right', prop=font1, frameon=False)  # 绘制图例，指定图例位置
plt.xticks([])  # 去掉x坐标轴刻度
plt.show()


def overlap(X, window_size, window_step):
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = (valid) // ss
    out = np.ndarray((nw, ws), dtype=a.dtype)

    for i in range(nw):
        # "slide" the window along the samples
        start = i * ss
        stop = start + ws
        out[i] = a[start: stop]

    return out


def stft(X, fftsize=128, step=65, mean_normalize=True, real=False,
         compute_onesided=True):
    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2
    if mean_normalize:
        X -= X.mean()

    X = overlap(X, fftsize, step)

    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def pretty_spectrogram(d, log=True, thresh=5, fft_size=512, step_size=64):
    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False,
                           compute_onesided=True))

    if log == True:
        specgram /= specgram.max()  # volume normalize to max 1
        specgram = np.log10(specgram)  # take log
        specgram[specgram < -thresh] = -thresh  # set anything less than the threshold as the threshold
    else:
        specgram[specgram < thresh] = thresh  # set anything less than the threshold as the threshold

    return specgram


### Parameters ###
fft_size = 1230  # window size for the FFT
step_size = int(fft_size / 16)  # distance to slide along the window (in time)
spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)

# Grab your wav and filter it


# mywav = '/Users/admin/Desktop/FA01_01.wav'
# rate, data = wavfile.read(mywav)
# data = butter_bandpass_filter(data, lowcut, highcut, rate, order=1)
# Only use a short clip for our demo
if np.shape(data)[0] / float(rate) > 10:
    data = data[0:rate * 10]
print('Length in time (s): ', np.shape(data)[0] / float(rate))

wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size=fft_size,
                                     step_size=step_size, log=True, thresh=spec_thresh)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
cax = ax.matshow(np.transpose(wav_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                 origin='lower')
fig.colorbar(cax)
plt.title('Original Spectrogram')
plt.show()

wav_mixed_spectrogram = pretty_spectrogram(mixed_data.astype('float64'), fft_size=fft_size,
                                           step_size=step_size, log=True, thresh=spec_thresh)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))
cax = ax.matshow(np.transpose(wav_mixed_spectrogram), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot,
                 origin='lower')
fig.colorbar(cax)
plt.title('Mixed Spectrogram')
plt.show()
