from __future__ import print_function

import tensorflow as tf
import numpy as np

m = np.reshape(np.arange(200), newshape=[5, 40])

z1 = tf.constant(m, dtype=tf.float32)

z2 = tf.constant(m + 1, dtype=tf.float32)

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_mean_std(x, axis=None, keepdims=False):

    mean = tf.reduce_mean(x, axis=axis, keepdims=keepdims)
    std = tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

    return mean, std


z_mean1 = tf.reduce_mean(z1, axis=[-1], keep_dims=True)

z_std1 = reduce_std(z1, axis=1, keepdims=True)

z_mean2 = tf.reduce_mean(z2, axis=[-1], keep_dims=True)

z_std2 = reduce_std(z2, axis=1, keepdims=True)

num = (z1 - z_mean1) * (z2 - z_mean2)
den = z_std1 * z_std2

auto_corr_loss = tf.reduce_mean(- num / den, name='auto-corr-loss')

with tf.Session() as sess:
    num_val = sess.run(num)

    # z_var1_val = sess.run(z_var1)
    # z_var2_val = sess.run(z_var2)
    den_val = sess.run(den)
    z_std1_val = sess.run(z_std1)
    auto_corr_loss_eval = sess.run(auto_corr_loss)

    # print(num_val)
    print(z_std1_val)
    # print(den_val)
    # print(z_std1_val)

    print(auto_corr_loss_eval)
