# 这次目的非常明确，得到spectrogram的数值矩阵


import matplotlib.pyplot as plt

import soundfile as sf
import scipy
import numpy as np
from scipy import signal
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# loading wav
signalData1, samplingFrequency1 = \
    sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/10001.flac')
signalData1 = signalData1[:16000 * 3]

print(len(signalData1))
print(signalData1)

f, t, Sxx = signal.spectrogram(signalData1, samplingFrequency1, mode='complex')
'''
print('This is f:')
print(f)
print(t)
print('This is Sxx:')
print(np.size(Sxx))#Sxx有69660个数据点
print(len(Sxx))#Sxx每行有129个数据，有540行，129*540=69660
print(np.size(t))#t有540个
print(len(t))
print(np.size(f))#f有129个
print(len(f))

print(Sxx)
'''
print(Sxx)
Sxx = np.abs(Sxx)
# levels = MaxNLocator(nbins=1000).tick_values(Sxx.min(), Sxx.max())
# cmap = plt.get_cmap('PiYG')
# norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
# plt.pcolor(t,f, Sxx, cmap=cmap, norm=norm)
f, ax = plt.subplots(figsize=(4.8, 2.4))
S = 20 * np.log10(Sxx / (np.max(Sxx)))
N = signalData1.shape[0]
L = N / samplingFrequency1

def MatrixMultiply(a, b):
    n=len(a)
    c=[[0]*n for row in range(n)] #初始化c为n行n列的全零矩阵
    for i in range(0, n):
        for j in range(0, n):
            #c[i][j]=0
            for k in range(0, n):
                c[i][j]=c[i][j]+a[i][k]*b[k][j]
    return c

Sxx2 = Sxx/20
S2 = np.where(S<-25,1,0)
S3 = MatrixMultiply(S, S2)

ax.imshow(S2, origin = 'lower',cmap='viridis',
          extent=(0,L,0,samplingFrequency1/2/1000))
ax.axis('tight')
plt.show()
# plt.colormaps(gray(100))
# colorbar
# plt.ylabel('Frequency [Hz]')
# # plt.ylim(0, 2000)
# plt.xlabel('Time [sec]')
# plt.show()

# plt.specgram(signalData1, Fs=samplingFrequency1)
# plt.show()

print(Sxx)

'/Users/admin/Desktop/FA01_01.wav'
