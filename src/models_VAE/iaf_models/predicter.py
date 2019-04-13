import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *


def predict(self, z):
    z_flat = tf.reshape(z, shape=[-1, self.hps.n_z * self.hps.n_z])

    y_predict = tf.contrib.layers.fully_connected(inputs=z_flat, num_outputs=self.hps.Tau * self.hps.C,
                                                  activation_fn=tf.nn.relu)

    y_predict = tf.reshape(y_predict, shape=[-1, self.hps.Tau, self.hps.C])
    return y_predict
