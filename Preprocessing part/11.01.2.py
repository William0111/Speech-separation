import matplotlib.pyplot as plt
import soundfile as sf
import scipy
import numpy as np
from scipy import signal

signalData1, samplingFrequency1  = \
    sf.read('/Users/admin/Desktop/FB07_01.wav')

samplingFrequency1 = int(samplingFrequency1)

Sxx, f, t, im = plt.specgram(signalData1, Fs = samplingFrequency1)
plt.ylim(0,15000)
plt.show()

#这个spectrogram的办法比较好

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
'''