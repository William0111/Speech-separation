# To have a first try by DNN

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# read data from files
def data(batch_size):
    for i in range(batch_size):
        data, rate = \
            sf.read('/Users/admin/Desktop/practiceunderpycharm/flac_dataforcnn/1000' + str(i) + '.flac')
        data = data[:16000 * 3]

    return data, rate


noise_power = np.var(data) * 2
noise = np.random.normal(scale=noise_power, size=data.shape)

mixed_data = data + noise

xs = tf.placeholder(tf.float32, [None, 16000 * 3]) / 255.  # 3 seconds
ys = tf.placeholder(tf.float32, [None, 16000 * 3])
keep_prob = tf.placeholder(tf.float32)

# the loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
