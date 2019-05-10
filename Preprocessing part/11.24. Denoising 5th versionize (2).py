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


### Structure part ###

# define placeholder for inputs to network

xs = tf.placeholder(tf.float32, [None, 2470])# , name="x_input")  # 247*10
ys = tf.placeholder(tf.float32, [None, 2470])# , name="y_input")

prediction = tf.placeholder(tf.float32, [None, 2470])
keep_prob = tf.placeholder(tf.float32)

## fc1 layer ##
with tf.name_scope('fully_connected_layer1'):
    W_fc1 = weight_variable([2470, 2470])
    b_fc1 = bias_variable([2470])
    h_fc1 = tf.nn.relu(tf.matmul(xs, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([2470, 2470])
b_fc2 = bias_variable([2470])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

## fc2 layer ##
W_fc3 = weight_variable([2470, 2470])
b_fc3 = bias_variable([2470])
prediction = tf.nn.sigmoid(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# the loss
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.reduce_sum((prediction - ys)))
    tf.summary.histogram('cross_entropy', cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

# -tf.reduce_sum(ys * tf.log(prediction),
#              reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(tf.global_variables_initializer())

clean_mask = np.loadtxt("clean_mask.txt")
spec_train = np.loadtxt("spec_train.txt")
clean_mask_test = np.loadtxt("clean_mask_test.txt")
spec_train_test = np.loadtxt("spec_train_test.txt")

for i in range(500):
    batch_xs = spec_train
    batch_ys = clean_mask

    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    # if i % 5 == 0:
    # a = tf.Print(prediction, [prediction])
    # print (sess.run(prediction))
    # result = a + 0
    if i % 10 == 0:
        # print(compute_accuracy(spec_train_test, clean_mask_test))
        #result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
        #writer.add_summary(result, i)
        print(compute_accuracy(spec_train_test, clean_mask_test))

# pre = sess.run(prediction,feed_dict={xs: batch_xs, ys: batch_ys})

# ### Check part
# a = stft(data, win, step)
# b = a.shape


# sess.run(prediction)
