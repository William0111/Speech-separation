

# import the pyplot and wavfile modules


import tensorflow as tf

import matplotlib.pyplot as plot
import soundfile as sf

#import pytorch as torch
from scipy.io import wavfile

# Read the wav file (mono)
asa = tf.ones(5)
print(asa)

signalData, samplingFrequency  = \
    sf.read('/Users/admin/Desktop/FA01_02.wav')

print(signalData)
print(samplingFrequency)

# Plot the signal read from wav file

plot.subplot(231)

plot.title('Spectrogram of a wav file1')

plot.plot(signalData)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

plot.subplot(234)

plot.specgram(signalData, Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')

plot.show()