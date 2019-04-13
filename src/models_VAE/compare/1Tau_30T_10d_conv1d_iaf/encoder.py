import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *


def encoder(self, x):
    is_training = get_current_tower_context().is_training

    # [M, T, D] => [M, T, f0]
    # fc_l1 = gaussian_dense(name='encode_l1', inputs=x, out_C=self.hps.f[0])
    conv_l1 = conv1d(name='encode_l1', inputs=x, kernel_size=self.hps.lst_kernels[0], in_C=self.hps.D,
                     out_C=self.hps.f[0])
    batch_l1 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l1)
    activate_l1 = tf.nn.relu(batch_l1)
    max_pool_l1 = tf.layers.max_pooling1d(inputs=activate_l1, pool_size=2, strides=1, padding='SAME',
                                          data_format='channels_last')

    out_l1 = tf.layers.dropout(inputs=max_pool_l1, rate=self.hps.dropout_rate, training=is_training)

    # [M, T, f0] => [M, T, f1]
    # fc_l2 = gaussian_dense(name='encode_l2', inputs=out_l1, out_C=self.hps.f[1])
    conv_l2 = conv1d(name='encode_l2', inputs=out_l1, kernel_size=self.hps.lst_kernels[1], in_C=self.hps.f[0],
                     out_C=self.hps.f[1])
    batch_l2 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l2)
    activate_l2 = tf.nn.tanh(batch_l2)
    max_pool_l2 = tf.layers.max_pooling1d(inputs=activate_l2, pool_size=2, strides=1, padding='SAME',
                                          data_format='channels_last')

    out_l2 = tf.layers.dropout(inputs=max_pool_l2, rate=self.hps.dropout_rate, training=is_training)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hps.lstm_units, state_is_tuple=True)
    # z: [M, T, o]
    # h: [M, o]
    # c: [M, o]
    # [M, T, f1] => [M, T, o]
    outputs, state = tf.nn.dynamic_rnn(cell, out_l2, sequence_length=[self.hps.T] * self.hps.M, dtype=tf.float32)
    # ,parallel_iterations=64)

    # [M, T, o] => [M, T * o] => [M, n_z]
    next_seq = tf.reshape(outputs, shape=[-1, self.hps.T * self.hps.lstm_units])
    state_c = state.c

    if self.hps.is_VDE:
        z_lst = gaussian_dense(name='encode_l3', inputs=next_seq, out_C=2 * self.hps.n_z * self.hps.n_z)
    else:
        rs_l3 = tf.reshape(out_l2, [-1, self.hps.T * self.hps.f[1]])
        z_lst = gaussian_dense(name='encode_l3', inputs=rs_l3, out_C=2 * self.hps.n_z * self.hps.n_z)

    z_lst = tf.reshape(z_lst, shape=[-1, self.hps.n_z, 2 * self.hps.n_z])
    z_mu, z_std1 = split(z_lst, split_dim=2, split_sizes=[self.hps.n_z, self.hps.n_z])

    z_std = 1e-10 + tf.nn.softplus(z_std1)

    if self.hps.is_VAE:
        noise = tf.random_normal(shape=tf.shape(z_mu), mean=0.0, stddev=1.0)
        z = z_mu + noise * z_std
    else:
        z = z_mu

    return z_mu, z_std, z, state_c
