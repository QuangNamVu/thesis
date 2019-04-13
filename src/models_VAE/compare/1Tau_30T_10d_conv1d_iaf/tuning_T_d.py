#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Ref: DCGAN.py Author: Yuxin Wu
# Author: Nam VQ


import argparse
import numpy as np
import os
import tensorflow as tf
from tensorpack.utils import logger
from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
# from tensorpack.tfutils import summary

from VDE import VDEModelDesc, VDETrainer  # , RandomZData
from info_params import get_default_hparams
from load_data import *
from tqdm import tqdm


# prefetch data
def get_data(hps):
    X_train_seq, X_test_seq, X_tau_train_seq, X_tau_test_seq, target_one_hot_train_seq, target_one_hot_test_seq = load_data_seq(
        hps)

    X_train_segment, X_tau_train_segment, y_one_hot_train_segment = segment_seq(X_train_seq, X_tau_train_seq, hps,
                                                                                target_one_hot=target_one_hot_train_seq)

    X_test_segment, X_tau_test_segment, y_one_hot_test_segment = segment_seq(X_test_seq, X_tau_test_seq, hps,
                                                                             target_one_hot=target_one_hot_test_seq)

    # X constructed = X tau
    if hps.is_VDE:
        train_data = LoadData(X_train_segment, X_tau_train_segment, y_one_hot_train_segment, shuffle=True)
        test_data = LoadData(X_test_segment, X_tau_test_segment, y_one_hot_test_segment, shuffle=False)

    else:
        # X constructed = X
        train_data = LoadData(X_train_segment, X_train_segment, y_one_hot_train_segment, shuffle=True)
        test_data = LoadData(X_test_segment, X_test_segment, y_one_hot_test_segment, shuffle=False)

    # ds_eps = RandomEpsilon(info_params.n_z)

    ds_train = ConcatData([train_data])
    ds_test = ConcatData([test_data])

    # ds = train_data
    ds_train = BatchData(ds_train, batch_size=hps.M)
    ds_test = BatchData(ds_test, batch_size=hps.M)

    return ds_train, ds_test


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


lst_T = list(range(1, 10, 1)) + list(range(10, 100, 10)) + list(range(100, 500, 50))
lst_D = list(range(1, 10, 1)) + list(range(10, 100, 10)) + list(range(100, 500, 50))


# lst_D = [1]

hps = get_default_hparams()
hps.steps_per_epoch = 4
hps.epochs = 500
hps.is_VAE = False
hps.is_VDE = True


args = get_args()

for t in tqdm(lst_T):
    for d in tqdm(lst_D):

        ckpt_dir = os.path.expanduser("~") + '/tuning/%dD_%dT' % (t, d)
        os.makedirs(ckpt_dir, exist_ok=True)
        args.logdir = ckpt_dir
        hps.lag_time = d
        hps.T = t
        M = VDEModelDesc(hps)
        ds_train, ds_test = get_data(hps)
        x = VDETrainer(
            input=QueueInput(ds_train), model=M).train_with_defaults(
            callbacks=[
                ModelSaver(checkpoint_dir=ckpt_dir),
                callbacks.MergeAllSummaries(),
                MinSaver('total_loss'),
                InferenceRunner(ds_test, [ScalarStats('predict_trend/accuracy_')])
            ],
            steps_per_epoch=hps.steps_per_epoch,
            max_epoch=hps.epochs,
            session_init=None
        )
        # tf.get_variable_scope().reuse_variables()
        tf.reset_default_graph()
        del M
        del ds_train
        del ds_test
        del x
