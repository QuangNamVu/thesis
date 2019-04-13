import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import *
from tf_utils.common import *


def build_losses(self, y_one_hot, x_hat):
    with tf.name_scope("encode"):
        log_post_z = -0.5 * tf.reduce_sum((np.log(2. * np.pi) + 1 + self.z_lsgm), 1)  # N: qz

        log_prior_z = -0.5 * tf.reduce_sum((np.log(2. * np.pi) + tf.exp(self.z_lsgm)), 1)  # N: pz

        # max elbo: => max(logpx + logpz - logqz)
        self.latent_loss = tf.reduce_mean((log_post_z - log_prior_z))  # - log_p_x)
        if self.hps.is_IAF:
            self.latent_loss = tf.subtract(self.latent_loss, tf.reduce_mean(self.z_lgsm_iaf), name='iaf_loss')

        add_moving_summary(self.latent_loss)
        if self.hps.is_IAF:
            z = tf.matrix_diag_part(self.z_iaf)
            z_tau = tf.matrix_diag_part(self.z_tau_iaf)
        else:
            z = tf.matrix_diag_part(self.z)
            z_tau = tf.matrix_diag_part(self.z_tau)

        if self.hps.is_VDE:
            self.auto_corr_loss = auto_corr_loss(z, z_tau, n_dim=2)
            add_moving_summary(self.auto_corr_loss)

        else:
            self.auto_corr_loss = tf.constant(0.0)

        if self.hps.check_error_z:
            # total_x_con = tf.reduce_mean(self.x_con, name='check_error')
            a, b = tf.nn.moments(self.z, axes=[1], keep_dims=False)
            tf.summary.scalar(name='check_z_mu', tensor=tf.reduce_mean(a))
            tf.summary.scalar(name='check_z_std', tensor=tf.reduce_mean(b))
            tf.summary.scalar(name='check_z_norm', tensor=tf.reduce_mean(tf.abs(self.z)))

    with tf.name_scope("predict_trend"):
        self.predict_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_one_hot, logits=self.y_pred)

        trend_labels_idx = tf.argmax(y_one_hot, axis=-1, name='trend_labels_idx')
        trend_labels_idx = tf.reshape(trend_labels_idx, shape=[-1])

        y_pred_idx = tf.argmax(self.y_pred, axis=-1, name='y_pred_idx')
        y_pred_idx = tf.reshape(y_pred_idx, shape=[-1])

        _, accuracy = tf.metrics.accuracy(labels=trend_labels_idx, predictions=y_pred_idx)

        # correct_prediction = tf.equal(tf.argmax(y_one_hot, -1), tf.argmax(self.y_pred, -1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.predict_accuracy = tf.identity(accuracy, name='accuracy_')

        tf.summary.scalar('prediction-accuracy', self.predict_accuracy)
        tf.summary.scalar('prediction-loss', self.predict_loss)

    with tf.name_scope("decode"):
        if self.hps.check_error_x_recon:
            # total_x_con = tf.reduce_mean(self.x_con, name='check_error')
            a, b = tf.nn.moments(self.x_con, axes=[1, 2], keep_dims=False)
            tf.summary.scalar(name='check_recon_mu', tensor=tf.reduce_mean(a))
            tf.summary.scalar(name='check_recon_std', tensor=tf.reduce_mean(b))
            tf.summary.scalar(name='check_recon_norm', tensor=tf.reduce_mean(tf.abs(self.x_con)))
        self.log_mse_loss = tf.log(tf.losses.mean_squared_error(labels=self.x_con, predictions=x_hat))
        tf.summary.scalar('log_mse', self.log_mse_loss)

        self.mse_loss = tf.losses.mean_squared_error(labels=self.x_con, predictions=x_hat)
        tf.summary.scalar('mse', self.mse_loss)

        # cross_entropy
        eps = 1e-10
        x_con_clip = tf.clip_by_value(self.x_con, eps, 1 - eps)
        x_hat_clip = tf.clip_by_value(x_hat, eps, 1 - eps)

        self.log_lik = tf.reduce_mean(-tf.reduce_sum(x_con_clip * tf.log(x_hat_clip), axis=1))

        tf.summary.scalar('log_likelihood_reconstructed', self.log_lik)

    self.total_loss = tf.add_n(
        [self.log_mse_loss, self.auto_corr_loss, 100 * self.predict_loss, self.latent_loss],
        # [self.log_mse_loss, self.predict_loss, self.latent_loss],
        name="total_cost")

    tf.summary.scalar('total-loss', self.total_loss)
