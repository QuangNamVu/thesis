import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *


def encoder(self, x):
    is_training = get_current_tower_context().is_training

    # [M, T, D] => [M, T, f0]
    conv_l1 = tf.layers.conv1d(inputs=x, filters=self.hps.f[0], kernel_size=self.hps.lst_kernels[0],
                               padding='same', data_format='channels_last')

    batch_l1 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l1, axis=-1)

    relu_l1 = tf.nn.relu(features=batch_l1)

    max_pool_l1 = tf.layers.max_pooling1d(inputs=relu_l1, pool_size=2, strides=1, padding='SAME',
                                          data_format='channels_last')

    # [M, T, f0] => [M, T, f1]
    conv_l2 = tf.layers.conv1d(inputs=max_pool_l1, filters=self.hps.f[1],
                               kernel_size=self.hps.lst_kernels[1], padding='same', data_format='channels_last')

    batch_l2 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l2, axis=-1)

    relu_l2 = tf.nn.relu(features=batch_l2)

    max_pool_l2 = tf.layers.max_pooling1d(inputs=relu_l2, pool_size=2, strides=1, padding='SAME',
                                          data_format='channels_last')

    dropout = tf.layers.dropout(inputs=max_pool_l2, rate=self.hps.dropout_rate, training=is_training)
    out_l2 = tf.reshape(dropout, [-1, self.hps.T * self.hps.f[1]])

    if self.hps.is_VDE:
        # [M, f[-1]] => [M, f[-1]//2, 2]

        z_3dims = tf.reshape(out_l2, shape=[-1, self.hps.T, self.hps.f[1]])

        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hps.lstm_units, state_is_tuple=True)

        # z: [M, f1//2, lstm_units]
        # h: [M, lstm_units]
        # c: [M, lstm_units]
        z_lstm, state = tf.nn.dynamic_rnn(cell, z_3dims, sequence_length=[self.hps.T] * self.hps.M, dtype=tf.float32)

        z_lst = tf.contrib.layers.fully_connected(inputs=state.h, num_outputs=3 * self.hps.n_z * self.hps.n_z,
                                                  activation_fn=tf.nn.relu)

    else:
        z_lst = tf.contrib.layers.fully_connected(inputs=out_l2, num_outputs=3 * self.hps.n_z * self.hps.n_z,
                                                  activation_fn=tf.nn.relu)

    z_mu, z_lsgm, lower_triag = split(z_lst, split_dim=1,
                                      split_sizes=[self.hps.n_z * self.hps.n_z, self.hps.n_z * self.hps.n_z,
                                                   self.hps.n_z * self.hps.n_z])

    z_mu = tf.reshape(z_mu, shape=[-1, self.hps.n_z, self.hps.n_z])

    z_lsgm = tf.reshape(z_lsgm, shape=[-1, self.hps.n_z, self.hps.n_z])

    l_prime = tf.reshape(lower_triag, shape=[-1, self.hps.n_z, self.hps.n_z])

    z_sigma = tf.exp(z_lsgm)

    l = tf.multiply(self.l_mask, l_prime) + tf.expand_dims(tf.matrix_diag_part(z_sigma), -1)

    epsilon = tf.random_normal((tf.shape(z_mu)), 0, 1)

    z = l * epsilon + z_mu

    return z_mu, z_lsgm, z
