import tensorflow as tf
from tensorpack import *
from tf_utils.ar_layers import *
from tf_utils.common import *
from tf_utils import common


def encoder(self, x):
    # l = conv1d(name='encode_l1', inputs=x, kernel_size=self.hps.k[0], stride=1,
    #            in_C=self.hps.D, out_C=self.hps.f[0])
    #
    l = gaussian_dense(name='encode_fc0', inputs=x, out_C=self.hps.f[0])

    # l = common.batch_norm(l, self.hps, origin_name='bn1')

    l = tf.reshape(l, shape=[-1, self.hps.T * self.hps.f[0]])

    z_lst = gaussian_dense(name='encode_fc1', inputs=l, out_C=2 * self.hps.n_z)
    # z_lst = tf.nn.leaky_relu(z_lst)
    z_mu, z_std = split(z_lst, split_dim=1, split_sizes=[self.hps.n_z, self.hps.n_z])
    z_std = tf.nn.softplus(z_std + tf.log(tf.exp(1.0) - 1))

    if self.hps.is_VAE:
        z = tf.contrib.distributions.MultivariateNormalDiag(loc=z_mu, scale_diag=z_std)
        z = z.sample()
    else:
        z = z_mu

    return z_mu, z_std, z
