
import os
import shutil
import torch as t
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

for i in range(10001,12303):
    file_name = '/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/'+str(i)+'.flac'
    signalData1, samplingFrequency1 = \
        sf.read(file_name)

    # print(samplingFrequency1)
    # print(len(signalData1))
    # signalData1 = signalData1[:16000*3]
    # print(len(signalData1))

    if len(signalData1) < 48000:
        print(file_name)
        #shutil.move(file_name, "/Users/admin/Desktop/untitled folder")


    # samplingFrequency1 = int(samplingFrequency1)
    #
    # Sxx, f, t, im = plt.specgram(signalData1, Fs = samplingFrequency1)
    # #plt.ylim(0,15000)
    # plt.show()