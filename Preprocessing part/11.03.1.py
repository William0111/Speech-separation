
#目的：找到可以直接学习的代码，利用mask的方法分离噪音
#加噪（1，找到噪声，加上去
#

import torch as t
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

signalData1, samplingFrequency1 = \
    sf.read('/Users/admin/Desktop/practiceunderpycharm/flac/2.flac')

samplingFrequency1 = int(samplingFrequency1)

Sxx, f, t, im = plt.specgram(signalData1, Fs = samplingFrequency1)
#plt.ylim(0,15000)
plt.show()



