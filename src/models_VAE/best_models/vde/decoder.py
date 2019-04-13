import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *


def decoder(self, z):
    is_training = get_current_tower_context().is_training
    fc_l1 = gaussian_dense(name='decode_fc1', inputs=z, out_C=self.hps.T * self.hps.f[1])
    activate_l1 = tf.nn.tanh(fc_l1)
    out_l1 = tf.reshape(activate_l1, shape=[-1, self.hps.T, self.hps.f[1]])

    # [M, T, f1] => [M, T, f0]
    fc_l2 = gaussian_dense(name='decode_l2', inputs=out_l1, out_C=self.hps.f[0])
    activate_l2 = tf.nn.elu(fc_l2)
    out_l1 = tf.layers.dropout(inputs=activate_l2, rate=self.hps.dropout_rate, training=is_training)

    # [M, T, f0] => [M, T, D]
    fc_l3 = gaussian_dense(name='decode_l3', inputs=out_l1, out_C=self.hps.D)
    out_l3 = tf.sigmoid(fc_l3)

    return out_l3
