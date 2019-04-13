import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *


def decoder(self, z):
    flat_l1 = tf.reshape(z, shape=[-1, self.hps.n_z * self.hps.n_z])

    fc_l1 = tf.contrib.layers.fully_connected(inputs=flat_l1, num_outputs=self.hps.T * self.hps.f[1],
                                              activation_fn=tf.nn.relu)

    rs_l1 = tf.reshape(fc_l1, shape=[-1, self.hps.T, self.hps.f[1]])

    conv_l2 = tf.layers.conv1d(inputs=rs_l1, filters=self.hps.f[0], kernel_size=self.hps.lst_kernels[1],
                               padding='same', data_format='channels_last')

    batch_l2 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l2, axis=-1)

    relu_l2 = tf.nn.relu(features=batch_l2)

    out_l2 = tf.layers.max_pooling1d(inputs=relu_l2, pool_size=2, strides=1, padding='SAME',
                                     data_format='channels_last')

    conv_l3 = tf.layers.conv1d(inputs=out_l2, filters=self.hps.D, kernel_size=self.hps.lst_kernels[0],
                               padding='same', data_format='channels_last')
    relu_l3 = tf.nn.relu(features=conv_l3)

    out_l3 = tf.layers.max_pooling1d(inputs=relu_l3, pool_size=2, strides=1, padding='SAME',
                                     data_format='channels_last')

    return out_l3
