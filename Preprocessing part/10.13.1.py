import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# import the pyplot and wavfile modules

from tinytag import TinyTag
tag = TinyTag.get('/Users/admin/Downloads/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac')

print('This track is by %s.' % tag.bitrate)
print('It is %f seconds long.' % tag.duration)
print(tag)

tag = TinyTag.get('/Users/admin/Downloads/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac', image=True)
image_data = tag.get_image()
print(image_data)
# time = tag.disc_total
#
# plt.plot(time,image_data)
# plt.show()


tag.album         # album as string
tag.albumartist   # album artist as string
tag.artist        # artist name as string
tag.audio_offset  # number of bytes before audio data begins
tag.bitrate       # bitrate in kBits/s
tag.disc          # disc number
tag.disc_total    # the total number of discs
tag.duration      # duration of the song in seconds
tag.filesize      # file size in bytes
tag.genre         # genre as string
tag.samplerate    # samples per second
tag.title         # title of the song
tag.track         # track number as string
tag.track_total   # total number of tracks as string
tag.year          # year or data as string

# import the pyplot and wavfile modules

import matplotlib.pyplot as plot

from scipy.io import wavfile
import soundfile as sf

# Read the wav file (mono)

samplingFrequency, signalData = sf.read('/Users/admin/Downloads/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac')

# Plot the signal read from wav file

plot.subplot(211)

plot.title('Spectrogram of a wav file with piano music')

plot.plot(signalData)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

plot.subplot(212)

plot.specgram(signalData, Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')

plot.show()





# import speech_recognition as sr
# r = sr.Recognizer()
#
#
#
# from scipy.io import wavfile
#
#
# samplingFrequency, signalData = sr.FlacFile('/Users/admin/Downloads/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac')
#
# # Plot the signal read from wav file
#
# plot.subplot(211)
#
# plot.title('Spectrogram of a wav file with piano music')
#
# plot.plot(signalData)
#
# plot.xlabel('Sample')
#
# plot.ylabel('Amplitude')
#
# plot.subplot(212)
#
# plot.specgram(signalData, Fs=samplingFrequency)
#
# plot.xlabel('Time')
#
# plot.ylabel('Frequency')
#
# plot.show()