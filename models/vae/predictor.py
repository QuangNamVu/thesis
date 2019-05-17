import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *


def predict(self, z):
    """
    :param self: model
    :param z ~ N(0, I) shape: [M, n_z] == [batch_size, n_z]
    :return: batch of matrix X shape: [M, tau, C] == [batch_size, next_seq_trend, num_classes]
    """
    l = gaussian_dense(name='fc', inputs=z, out_C=self.hps.T * self.hps.f[0])
    l = tf.reshape(l, shape=[-1, self.hps.T, self.hps.f[0]])

    l = conv1d(name='conv_l1', inputs=l, kernel_size=5, stride=1,
               in_C=self.hps.f[0], out_C=128)

    l = tf.layers.batch_normalization(inputs=l)
    l = tf.nn.elu(l)

    l = conv1d(name='conv_l2', inputs=l, kernel_size=5, stride=1,
               in_C=128, out_C=64)

    l = tf.layers.batch_normalization(inputs=l)
    l = tf.nn.elu(l)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hps.lstm_units, state_is_tuple=True)  # , activation=tf.nn.tanh)
    outputs, state = tf.nn.dynamic_rnn(cell, l, sequence_length=[self.hps.T] * self.hps.M, dtype=tf.float32)

    x_con = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=self.hps.D)

    if self.hps.normalize_data in ['min_max', 'min_max_centralize']:
        x_con = tf.nn.sigmoid(x_con)

    fc_l1 = tf.contrib.layers.fully_connected(inputs=state.c, num_outputs=self.hps.Tau * self.hps.C)

    y_predict = tf.reshape(fc_l1, shape=[-1, self.hps.Tau, self.hps.C])
    y_predict = tf.sigmoid(y_predict) / tf.reduce_sum(tf.sigmoid(y_predict), axis=[-1], keepdims=True)
    return x_con, y_predict
