

# import the pyplot and wavfile modules

#Mixed wav files no matter the difference they are


import matplotlib.pyplot as plot
import soundfile as sf

#import pytorch as torch
from scipy.io import wavfile

# Read the wav file (mono)


signalData1, samplingFrequency1  = \
    sf.read('/Users/admin/Desktop/FB07_01.wav')

signalData2, samplingFrequency2  = \
    sf.read('/Users/admin/Desktop/MC13_01.wav')

print(len(signalData1))
print(len(signalData2))


if len(signalData2) < len(signalData1):
    signalData1 = signalData1[:len(signalData2)]
else:
    signalData2 = signalData2[:len(signalData1)]

print(signalData1)
print(samplingFrequency1)
print(len(signalData1))


print(signalData2)
print(samplingFrequency2)
print(len(signalData2))

mix_data = signalData1 + signalData2
print(mix_data)

# Plot the signal read from wav file

plot.subplot(321)

plot.title('wav file1')

plot.plot(signalData1)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

plot.subplot(322)

plot.title('Spectrogram of a wav file1')

plot.specgram(signalData1, Fs=samplingFrequency1)
######
plot.subplot(323)

plot.title('wav file2')

plot.plot(signalData2)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

plot.subplot(324)

plot.title('Spectrogram of a wav file2')

plot.specgram(signalData2, Fs=samplingFrequency2)
#######
plot.subplot(325)

plot.title('the mixed wav file')

plot.plot(mix_data)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

plot.subplot(326)

plot.title('Spectrogram of the mixed wav file')

plot.specgram(mix_data, Fs=samplingFrequency1)





plot.xlabel('Time')

plot.ylabel('Frequency')

plot.show()