import numpy as np
import tensorflow as tf


def gaussian_diag_logps(mean, logvar, sample=None):
    if sample is None:
        noise = tf.random_normal(tf.shape(mean))
        sample = mean + tf.exp(0.5 * logvar) * noise

    return -0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) / tf.exp(logvar))


class DiagonalGaussian(object):

    def __init__(self, mean, logvar, sample=None):
        self.mean = mean
        self.logvar = logvar

        if sample is None:
            noise = tf.random_normal(tf.shape(mean))
            sample = mean + tf.exp(0.5 * logvar) * noise
        self.sample = sample

    def logps(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)


def logsumexp(x):
    x_max = tf.reduce_max(x, [1], keep_dims=True)
    return tf.reshape(x_max, [-1]) + tf.log(tf.reduce_sum(tf.exp(x - x_max), [1]))


def compute_lowerbound(log_pxz, sum_kl_costs, k=1):
    if k == 1:
        return sum_kl_costs - log_pxz

    # log 1/k \sum p(x | z) * p(z) / q(z | x) = -log(k) + logsumexp(log p(x|z) + log p(z) - log q(z|x))
    log_pxz = tf.reshape(log_pxz, [-1, k])
    sum_kl_costs = tf.reshape(sum_kl_costs, [-1, k])
    return - (- tf.log(float(k)) + logsumexp(log_pxz - sum_kl_costs))


def cross_entropy_loss(p, q, is_clip_q=False):
    if is_clip_q:
        q = tf.clip_by_value(q, clip_value_min=1e-10, clip_value_max=1 - 1e-10)

    return -tf.reduce_mean(p * tf.log(q) + (1 - p) * tf.log(1 - q))
