# -*- coding: utf-8 -*-
# Ref: Yuxin Wu file GAN.py
# Author: VQNam
import tensorflow as tf
from tensorpack import *
from tensorpack import ModelDescBase, StagingInput, TowerTrainer
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.utils.argtools import memoized_method

# from utils import *
from encoder import encoder
from decoder import decoder
from predictor import predict
from tf_utils.multi_iaf import multi_iaf
from build_loss import build_losses


class ModelDesc(ModelDescBase):
    def __init__(self, hps):
        self.hps = hps

    def inputs(self):
        return [
            tf.TensorSpec((None, self.hps.T, self.hps.D), tf.float32, 'x'),
            tf.TensorSpec((None, self.hps.T, self.hps.D), tf.float32, 'x_next_target'),
            tf.TensorSpec((None, self.hps.Tau, self.hps.C), tf.float32, 'y_one_hot')
        ]

    def collect_variables(self, encode_scope='encode', predict_scope='predict', decode_scope='decode'):
        self.encode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, encode_scope)
        assert self.encode_vars, "Encode graph not found"

        self.predict_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, predict_scope)
        assert self.predict_vars, "Predict graph not found"

        self.decode_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, decode_scope)
        assert self.decode_vars, "Decode graph not found"

    def encoder(self, x):
        return encoder(self, x)

    def predict(self, z):
        return predict(self, z)

    def decoder(self, z):
        return decoder(self, z)

    def multi_iaf(self, z):
        return multi_iaf(self, z)

    def build_losses(self, y_one_hot, x, x_next):
        build_losses(self, y_one_hot, x, x_next)

    def build_graph(self, x, x_next_target, y_one_hot):
        with tf.variable_scope('encode', reuse=False):
            self.z_mu, self.std, self.z = self.encoder(x)
        with tf.variable_scope('decode', reuse=False):
            self.x_con = self.decoder(self.z)
        with tf.variable_scope('predict', reuse=False):
            self.x_next, self.y_pred = self.predict(self.z)
        with tf.variable_scope('encode', reuse=True):
            _, _, self.z_tau = self.encoder(self.x_next)
            # self.z_tau = self.z_tau_mu + noise * tf.exp(0.5 * self.z_tau_lsgms)

        self.build_losses(y_one_hot=y_one_hot, x=x, x_next=x_next_target)

        self.collect_variables()

    def optimizer(self):
        optimizer_origin = tf.train.AdamOptimizer(learning_rate=self.hps.learning_rate)

        return tf.contrib.estimator.clip_gradients_by_norm(optimizer_origin, clip_norm=1.0)

    @memoized_method
    def get_optimizer(self):
        return self.optimizer()


class Trainer(TowerTrainer):

    def __init__(self, input, model, num_gpu=1):
        """
        Args:
            input (InputSource):
            model (VDEModelDesc):
        """
        super(Trainer, self).__init__()
        assert isinstance(model, ModelDesc), model

        if num_gpu > 1:
            input = StagingInput(input)

        # Setup input
        cbs = input.setup(model.get_input_signature())
        self.register_callback(cbs)

        assert num_gpu <= 1, "Should be 1 gpu for small data"

        self._build_vde_trainer(input, model)

    def _build_vde_trainer(self, input, model):
        """
        Args:
            input (InputSource):
            model (VDEModelDesc):
        """
        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.get_input_signature())
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())
        opt = model.get_optimizer()

        with tf.name_scope('optimize'):
            vde_min = opt.minimize(model.total_cost,
                                   var_list=[model.encode_vars, model.predict_vars, model.decode_vars], name='train_op')
        self.train_op = vde_min
