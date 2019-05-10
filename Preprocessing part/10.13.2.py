# import the pyplot and wavfile modules

import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
from scipy.io import wavfile

# Read the wav file (mono)

samplingFrequency, signalData = sf.read('/Users/admin/Downloads/LibriSpeech/dev-clean/8842/304647/8842-304647-0002.flac')

# Frequency000 = samplingFrequency
#
# print(samplingFrequency)
# print(signalData)

print(sf.read('/Users/admin/Downloads/LibriSpeech/dev-clean/8842/304647/8842-304647-0004.flac'))

# plot.subplot(211)
# plot.plot(signalData)
# plot.subplot(212)
# plot.plot(samplingFrequency)



x = samplingFrequency
#scipy.signal.spectrogram(x, fs=1.0, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
fs = 10000000
f, t, Sxx = signal.spectrogram(x, fs)

plt.pcolormesh(t, f, Sxx)
plt.xlim((0, 0.05))
plt.ylim((20000, 70000))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


# Plot the signal read from wav file

# plot.subplot(211)
#
# plot.title('Spectrogram of a flac file with piano music')
#
# plot.plot(signalData)
#
# plot.xlabel('Sample')
#
# plot.ylabel('Amplitude')

# plot.subplot(212)
#
# plot.specgram(signalData, Fs=samplingFrequency)
#
# plot.xlabel('Time')
#
# plot.ylabel('Frequency')

#plot.show()