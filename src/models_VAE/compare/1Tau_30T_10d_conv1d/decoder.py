import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *


def decoder(self, z):
    is_training = get_current_tower_context().is_training
    # [M, n_z, n_z] => [M, n_z, T]
    fc_l1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=self.hps.T, activation_fn=tf.nn.relu)

    # [M, n_z, T] => [M, T, n_z] => [M, T, f1]
    rs_l1 = tf.reshape(fc_l1, shape=[-1, self.hps.T, self.hps.n_z])
    conv_l1 = conv1d(name='decode_l2', inputs=rs_l1, kernel_size=1, in_C=self.hps.n_z, out_C=self.hps.f[1])
    activation_l1 = tf.nn.tanh(conv_l1)
    out_l1 = tf.layers.max_pooling1d(inputs=activation_l1, pool_size=2, strides=1, padding='SAME',
                                     data_format='channels_last')

    # [M, T, f1] => [M, T, f0]
    conv_l2 = conv1d(name='decode_l3', inputs=out_l1, kernel_size=1, in_C=self.hps.f[1], out_C=self.hps.f[0])
    activation_l2 = tf.nn.relu(conv_l2)
    out_l2 = tf.layers.max_pooling1d(inputs=activation_l2, pool_size=2, strides=1, padding='SAME',
                                     data_format='channels_last')

    conv_l3 = conv1d(name='decode_l4', inputs=out_l2, kernel_size=1, in_C=self.hps.f[0], out_C=self.hps.D)
    out_l3 = tf.sigmoid(conv_l3)

    return out_l3
