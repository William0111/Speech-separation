# The aim is to use DNN or RNN to learn denoising
# The difficulty might come across will be the input data into train file or test file

import IPython.display
from ipywidgets import interact, interactive, fixed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import copy
from scipy.fftpack import fft
from scipy import ifft
from scipy.signal import butter, lfilter
import scipy.ndimage
import soundfile as sf
from scipy import ceil, complex64, float64, hamming, zeros
import tensorflow as tf


### Function part ###

# read file
def read_Flac(filename):
    data, fs = \
        sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/' + str(filename) + '.flac')
    return data, fs


# get spectrogram
def stft(x, win, step):
    l = len(x)  # length of data
    N = len(win)  # length of window
    M = int(ceil(float(l - N + step) / step))  # Number of Windows in the spectrogram

    new_x = zeros(N + ((M - 1) * step), dtype=float64)
    new_x[: l] = x

    X = zeros([M, N], dtype=complex64)  # Initialization of spectrogram (complex type)
    for m in range(M):
        start = step * m
        X[m, :] = fft(new_x[start: start + N] * win)
    return X


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def get_mask(spectrogram):
    s = spectrogram
    s = np.where(abs(s) > 1, s, 0)
    s = np.where(abs(s) < 1, s, 1)
    s = np.real(s)
    # masked_signal_spectrogram = np.logical_and(spectrogram, s) * spectrogram
    return s  # masked_signal_spectrogram


###  Global Parameter part ###
data = read_Flac(10005)[0]
fs = read_Flac(10005)[1]
data = data[:16000 * 2]  # must read data befor this step, so can cut into same length
fftLen = 512
win = hamming(fftLen)
step = int(fftLen / 4)

### Structure part ###

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 4940])  # 247*512
ys = tf.placeholder(tf.float32, [None, 4940])
keep_prob = tf.placeholder(tf.float32)

## fc1 layer ##
W_fc1 = weight_variable([4940, 4940])
b_fc1 = bias_variable([4940])
h_fc1 = tf.nn.relu(tf.matmul(xs, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([4940, 4940])
b_fc2 = bias_variable([4940])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

## fc2 layer ##
W_fc3 = weight_variable([4940, 4940])
b_fc3 = bias_variable([4940])
prediction = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# the loss
cross_entropy = tf.reduce_mean(tf.square(prediction - ys))

#-tf.reduce_sum(ys * tf.log(prediction),
                                #              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

j = 0
clean_mask = np.zeros(4940)
spec_train = np.zeros(4940)
while j < 100:
    x1 = read_Flac(10001 + j)[0]
    x1 = x1[:16000 * 2]

    noise_power = np.var(x1) * 5
    noise = np.random.normal(scale=noise_power, size=x1.shape)
    x2 = x1 + noise

    x1_spec = abs(stft(x1, win, step))
    x1_spec = get_mask(x1_spec).reshape(247 * 512)
    x1_spec = x1_spec[247 * 490:247 * 510]
    # x1_spec = x1_spec / max(x1_spec)
    clean_mask = np.vstack((clean_mask, x1_spec))

    x2_spec = abs(stft(x2, win, step).reshape(247 * 512))

    x2_spec = x2_spec[247 * 490:247 * 510]
    x2_spec = x2_spec  # / max(x2_spec)
    spec_train = np.vstack((spec_train, x2_spec))

    j = j + 1

i = 0
clean_mask_test = np.zeros(4940)
spec_train_test = np.zeros(4940)
while i < 100:
    x1 = read_Flac(11001 + i)[0]
    x1 = x1[:16000 * 2]

    noise_power = np.var(x1) * 5
    noise = np.random.normal(scale=noise_power, size=x1.shape)
    x2 = x1 + noise

    x1_spec = abs(stft(x1, win, step))
    x1_spec = get_mask(x1_spec).reshape(247 * 512)
    x1_spec = x1_spec[247 * 490:247 * 510]
    # x1_spec = x1_spec / max(x1_spec)
    clean_mask_test = np.vstack((clean_mask_test, x1_spec))

    x2_spec = abs(stft(x2, win, step).reshape(247 * 512))
    # x2_spec = get_mask(x2_spec)
    x2_spec = x2_spec[247 * 490:247 * 510]
    x2_spec = x2_spec  # / max(x2_spec)
    spec_train_test = np.vstack((spec_train_test, x2_spec))

    i = i + 1

clean_mask = np.delete(clean_mask, 0, 0)
spec_train = np.delete(spec_train, 0, 0)
clean_mask_test = np.delete(clean_mask_test, 0, 0)
spec_train_test = np.delete(spec_train_test, 0, 0)

for i in range(1000):
    batch_xs = spec_train
    batch_ys = clean_mask

    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.8})
    # if i % 5 == 0:
    # a = tf.Print(prediction, [prediction])
    # print (sess.run(prediction))
    # result = a + 0
    if i % 10 == 0:
        print(compute_accuracy(
            spec_train_test, clean_mask_test))

# ### Check part
# a = stft(data, win, step)
# b = a.shape


# sess.run(prediction)
