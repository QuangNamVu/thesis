import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.summary import *
from tf_utils.common import *


# from tf_utils.summary_tools import *


# import tensorflow_probability as tfp

def build_losses(self, y_one_hot, x, x_next_target):
    with tf.name_scope("encode"):
        if self.hps.is_VAE:

            latent_loss = 0.5 * tf.reduce_mean(tf.square(self.z_mu) + tf.square(self.std) -
                                               tf.log(tf.square(self.std)) - 1.0, axis=1)

            # p = tf.sigmoid(self.z)
            # latent_loss = -tf.reduce_mean(p * tf.log(p) + (1 - p) * tf.log(1 - p), axis=1)

            # # max elbo: => max(logpx + logpz - logqz)
            # log_post_z = -0.5 * tf.reduce_sum((np.log(2. * np.pi) + 1 + tf.log(self.std)), 1)  # N: qz
            # log_prior_z = -0.5 * tf.reduce_sum((np.log(2. * np.pi) + self.std), 1)  # N: pz
            # latent_loss = tf.reduce_mean((log_post_z - log_prior_z))  # - log_p_x)

            self.latent_loss = tf.reduce_mean(latent_loss, name='latent_loss')

            if self.hps.is_IAF:
                self.latent_loss = tf.subtract(self.latent_loss, tf.reduce_mean(self.z_lgsm_iaf), name='iaf_loss')
        else:
            self.latent_loss = tf.constant(0.0)

        add_moving_summary(self.latent_loss)
        if self.hps.is_IAF:
            self.z = self.z_iaf
            self.z_tau = self.z_tau_iaf
            # z_mean, z_std = reduce_mean_std(self.z_iaf, axis=[1], keepdims=True)
            # z_tau_mean, z_tau_std = reduce_mean_std(self.z_tau_iaf, axis=[1], keepdims=True)

        if self.hps.is_VDE:
            z_mean, z_std = reduce_mean_std(self.z, axis=[1], keepdims=True)
            z_tau_mean, z_tau_std = reduce_mean_std(self.z_tau, axis=[1], keepdims=True)

            num = (self.z - z_mean) * (self.z_tau - z_tau_mean)
            den = z_std * z_tau_std

            self.auto_corr_loss = tf.reduce_mean(- tf.truediv(num, den), name='auto-corr-loss')
            add_moving_summary(self.auto_corr_loss)

        else:
            self.auto_corr_loss = tf.constant(0.0)

        if self.hps.check_error_z:
            tf.summary.scalar(name='check_z_std', tensor=tf.reduce_mean(tf.abs(self.std)))
            tf.summary.scalar(name='check_z_mu', tensor=tf.reduce_mean(tf.abs(self.z_mu)))
            tf.summary.scalar(name='check_z_norm', tensor=tf.reduce_mean(tf.abs(self.z)))

    with tf.name_scope("predict_trend"):
        self.teacher_force_loss = tf.losses.mean_squared_error(labels=x_next_target, predictions=self.x_next)
        tf.summary.scalar('teacher-force-loss', self.teacher_force_loss)

        # self.predict_loss = tf.losses.softmax_cross_entropy(onehot_labels=y_one_hot,
        #                                                     logits=self.y_pred)

        predict_loss = -tf.reduce_sum(y_one_hot * tf.log(self.y_pred), axis=-1)
        self.predict_loss = tf.reduce_mean(predict_loss)

        trend_labels_idx = tf.argmax(y_one_hot, axis=-1, name='trend_labels_idx')
        y_pred_idx = tf.argmax(self.y_pred, axis=-1, name='y_pred_idx')

        # y_pred_one_hot = tf.one_hot(y_pred_idx, depth=self.hps.C, name='y_pred_one_hot')
        _, accuracy = tf.metrics.accuracy(labels=trend_labels_idx, predictions=y_pred_idx)

        # img_d_summary = plot_confusion_matrix(correct_labels, predict_labels, labels, tensor_name='Summary/cm')
        # tf.add_summary(img_d_summary)
        cm = tf.confusion_matrix(
            labels=tf.reshape(trend_labels_idx, shape=[-1]),
            predictions=tf.reshape(y_pred_idx, shape=[-1]),
            dtype=tf.float32
        )

        tf.summary.image(
            name="Confusion_matrix",
            tensor=tf.expand_dims(tf.expand_dims(cm, -1), 0),
            max_outputs=3,
        )
        # correct_prediction = tf.equal(tf.argmax(y_one_hot, -1), tf.argmax(self.y_pred, -1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.predict_accuracy = tf.identity(accuracy, name='accuracy_')

        # tf.summary.scalar('prediction-accuracy', self.predict_accuracy)
        # tf.summary.scalar('prediction-loss', self.predict_loss)
        add_moving_summary(self.predict_accuracy)

    with tf.name_scope("decode"):
        if self.hps.check_error_x_recon:
            # total_x_con = tf.reduce_mean(self.x_con, name='check_error')
            a, b = tf.nn.moments(self.x_con, axes=[1, 2], keep_dims=False)
            tf.summary.scalar(name='check_recon_mu', tensor=tf.reduce_mean(a))
            tf.summary.scalar(name='check_recon_std', tensor=tf.reduce_mean(b))
            tf.summary.scalar(name='check_recon_norm', tensor=tf.reduce_mean(tf.abs(self.x_con)))

        if self.hps.normalize_data in ['min_max', 'min_max_centralize']:
            # cross_entropy
            eps = 1e-10
            x_con_clip = tf.clip_by_value(self.x_con, eps, 1 - eps)

            self.log_lik_loss = - tf.reduce_mean(
                x * tf.log(1 - x_con_clip) + (1 - x) * tf.log(1 - x_con_clip))

            tf.summary.scalar('log_likelihood_reconstructed', self.log_lik_loss)

            self.px_loss = self.log_lik_loss
        elif self.hps.normalize_data is 'z_score':
            self.mse_loss = tf.losses.mean_squared_error(labels=x, predictions=self.x_con)
            self.log_mse_loss = tf.log(self.mse_loss)
            self.rmse_loss = tf.sqrt(self.mse_loss)
            tf.summary.scalar('mse', self.mse_loss)
            tf.summary.scalar('log_mse', self.log_mse_loss)
            tf.summary.scalar('rmse', self.rmse_loss)
            self.px_loss = self.log_mse_loss

    with tf.name_scope("regularization"):
        if self.hps.l2_loss_eta is not 0.0:
            vars = tf.trainable_variables()
            self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * self.hps.l2_loss_eta

        else:
            self.l2_loss = 0.0

    self.total_cost = tf.add_n(
        # [self.hps.alpha * self.px_loss, 0.05 * self.auto_corr_loss, self.hps.beta * self.predict_loss,
        #  self.hps.alpha * self.teacher_force_loss, self.hps.gamma * self.latent_loss, self.l2_loss], name="total_cost")
        [self.hps.alpha * self.px_loss, self.auto_corr_loss, self.hps.beta * self.predict_loss, self.hps.gamma * self.latent_loss, self.l2_loss], name="total_cost")

    tf.summary.scalar('total-cost', self.total_cost)
