import tensorflow as tf
from tensorpack import *
from tensorpack.models.pool import UnPooling2x2ZeroFilled
from tf_utils.ar_layers import *
from tf_utils.common import *


def decoder(self, z):
    l = inverse_leaky_relu(z)
    l = gaussian_dense(name='fc', inputs=l, out_C=self.hps.T * self.hps.f[0])

    l = tf.reshape(l, shape=[-1, self.hps.T, self.hps.f[0]])

    # l = inverse_batch_norm(l, self.hps, origin_name='encode/bn1')

    # l = inverse_conv1d(name='l1', M=self.hps.M, T=self.hps.T, k=self.hps.k[0], stride=1,
    #                    in_C=self.hps.f[1], out_C=self.hps.f[0], value=l)

    # l = inverse_conv1d(name='l0', M=self.hps.M, T=self.hps.T, k=self.hps.k[0], stride=1,
    #                    in_C=self.hps.f[0], out_C=self.hps.D, value=l)

    l = gaussian_dense(name='l0', inputs=l, out_C=self.hps.D)

    if self.hps.normalize_data in ['min_max', 'min_max_centralize']:
        l = tf.nn.sigmoid(l)

    return l
