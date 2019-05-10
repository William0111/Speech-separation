#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 05:22:25 2019

@author: william
"""

import numpy as np

X_train = np.loadtxt("1209.2_spec_train_1000.txt")

x_c = np.loadtxt("clear_spec_input_250.txt")
x_c_spec_add = x_c[:1]


#
# def split_Spectro(name_spectro):
#     x_c_spec = np.zeros(1536)
#     i = 0
#     while i < x_c[0]:
#         x_c_spec_add = x_c[]
#
#
#
#     while j < number:
#         x1 = read_Flac(start_from + j)[0]
#         x1 = x1[:16000 * 2]
#
#         noise_with_power = np.var(x1) * noise_power
#         noise = np.random.normal(scale=noise_with_power, size=x1.shape)
#         x1 = x1 + noise
#
#         x1_spec = abs(stft(x1, win, step)).reshape(247 * 512)
#         high_boundary = 247 * 500
#         low_boundary = 247 * (500 - length)
#         x1_spec = x1_spec[low_boundary:high_boundary]  # cut between low boundry and high boundry
#         x1_spec = x1_spec / np.max(x1_spec)
#         spec = np.vstack((spec, x1_spec))
#
#         j = j + 1
#     spec = np.delete(spec, 0, 0)
#     return spec
