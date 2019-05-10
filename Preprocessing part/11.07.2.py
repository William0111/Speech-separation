# get complex spectrogram
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

signalData1, samplingFrequency1 = \
    sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/10001.flac')

signalData1 = signalData1[:16000 * 3]

# Sxx, f, t, im = plt.specgram(signalData1, Fs=samplingFrequency1)


N = signalData1.shape[0]
L = N / samplingFrequency1

t = np.arange(N) / samplingFrequency1

from skimage import util

M = 1024

slices = util.view_as_windows(signalData1, window_shape=(M,), step=100)
print(slices)

win = np.hanning(M + 1)[:-1]
slices = slices * win
print(slices)

slices = slices.T
print(slices)
print(slices.shape)

spectrum = np.fft.fft(slices, axis=0)[:M // 2 + 1:-1]
print(spectrum)

# fig, ax = plt.subplots(figsize=(4.8, 2.4))
#
# ax.plot(t, signalData1)
