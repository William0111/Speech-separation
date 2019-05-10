# try denoising

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

signalData1, samplingFrequency1 = \
    sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/10001.flac')

# print(samplingFrequency1)
# print(len(signalData1))
#signalData1 = signalData1[:16000 * 3]
# print(len(signalData1))

samplingFrequency1 = int(samplingFrequency1)

Sxx, f, t, im = plt.specgram(signalData1, Fs=samplingFrequency1, Fc=0, detrend=mlab.detrend_none,
                             window=mlab.window_hanning, noverlap=128,
                             cmap=None, xextent=None, pad_to=None, sides='default',
                             scale_by_freq=None, scale='default',
                            )

plt.show()

print(Sxx)
print(f)
print(t)
