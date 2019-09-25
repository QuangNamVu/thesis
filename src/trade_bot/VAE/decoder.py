import tensorflow as tf
from tensorpack import *
from .tf_utils.ar_layers import *
from .tf_utils.common import *


def decoder(self, z):
    is_training = get_current_tower_context().is_training
    fc_l1 = gaussian_dense(name='decode_fc1', inputs=z, out_C=self.hps.T * self.hps.f[1])
    activate_l1 = tf.nn.tanh(fc_l1)
    out_l1 = tf.reshape(activate_l1, shape=[-1, self.hps.T, self.hps.f[1]])

    # [M, T, f1] => [M, T, f0]
    conv_l2 = conv1d(name='decode_l2', inputs=out_l1, kernel_size=self.hps.lst_kernels[1], in_C=self.hps.f[1],
                     out_C=self.hps.f[0])
    batch_l2 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l2)
    activate_l2 = tf.nn.tanh(batch_l2)
    max_pool_l2 = tf.layers.max_pooling1d(inputs=activate_l2, pool_size=2, strides=1, padding='SAME',
                                          data_format='channels_last')
    out_l2 = tf.layers.dropout(inputs=max_pool_l2, rate=self.hps.dropout_rate, training=is_training)

    # [M, T, f0] => [M, T, D]
    conv_l3 = conv1d(name='decode_l3', inputs=out_l2, kernel_size=self.hps.lst_kernels[0],
                     in_C=self.hps.f[0], out_C=self.hps.D)
    batch_l3 = tf.layers.batch_normalization(momentum=self.hps.batch_norm_moment, inputs=conv_l3)
    out_l3 = tf.nn.elu(batch_l3)
    # max_pool_l3 = tf.layers.max_pooling1d(inputs=activate_l3, pool_size=2, strides=1, padding='SAME',
    #                                       data_format='channels_last')
    # out_l3 = tf.layers.dropout(inputs=max_pool_l3, rate=self.hps.dropout_rate, training=is_training)

    return out_l3
