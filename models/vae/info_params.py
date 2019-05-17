import os
import math
from tf_utils.hparams import *

home_path = os.path.expanduser("~") + '/'

# regularization convolution
IS_CONV_l2 = False

# data_file_name = home_path + 'data/cryptodatadownload/moving_average_240h.csv'
data_file_name = home_path + "data/cryptodatadownload/moving_average_240h.csv"
attributes_normalize_mean = ['Close', 'Volume BTC', 'Spread High-Low', 'Spread Close-Open', "MA_Close", "MA_V_BTC"]

is_concat = False

# attributes_normalize_mean = ['Close', 'Volume BTC']
N_train_seq = 10000
# N_train_seq = 4900

# date_fmt = '%Y-%m-%d %H-%p'
date_fmt = '%Y-%m-%d %I-%p'

# data_file_name = home_path + 'data/bitcoin-historical-data/small.csv'
# attributes_normalize_mean = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)',
#                              'Weighted_Price']
attributes_normalize_log = []
D = len(attributes_normalize_log + attributes_normalize_mean)
if is_concat:
    D *= 2

is_differencing = True
lag_time = 1  # X10 - X0 ;; X11 - X1

normalize_data = 'z_score'
# normalize_data = 'min_max'
# normalize_data = 'min_max_centralize'
# normalize_data = 'default'
# 'z_score' 'min_max' 'default' 'min_max_centralize'
normalize_data_idx = True

is_VAE = True
is_VDE = True
is_IAF = False
# normalize_data_idx = False

alpha = 4
beta = 7
gamma = 16
T = 120
Tau = 1
next_shift_VDE = 0
n_z = 120
k = [5, 3, 3, 3, 3]
# f = [32, 32, 64, 64, 4 * n_z]
f = [6, 64, 32, 4 * n_z]
l2_loss_eta = 1e-3
# lst_T = [60, 60, 30, 24]
t = [T, T // 2, T // 2, T // 4]
# n_z = int(T / k)


check_error_x_recon = True
check_error_z = True
lst_kernels = [11]
M = 1024  # Batch size
l2_loss = 1e-6

lstm_units = 100
C = 2
steps_per_epoch = 16
epochs = 900000

lst_kernels_iaf = [3, 3]

batch_norm_moment = .99

dropout_rate = .4


def get_default_hparams():
    return HParams(

        data_file_name=data_file_name,
        date_fmt=date_fmt,
        is_differencing=is_differencing,
        normalize_data=normalize_data,
        t=t,
        normalize_data_idx=normalize_data_idx,

        is_VAE=is_VAE,
        is_VDE=is_VDE,
        is_IAF=is_IAF,

        alpha=alpha,
        beta=beta,
        gamma=gamma,

        attributes_normalize_mean=attributes_normalize_mean,
        attributes_normalize_log=attributes_normalize_log,
        T=T,
        lag_time=lag_time,
        next_shift_VDE=next_shift_VDE,
        M=M,
        N_train_seq=N_train_seq,

        learning_rate=1e-3,

        l2_loss=l2_loss,  # l2 regularization
        is_concat=is_concat,
        D=D,

        n_z=n_z,
        lstm_units=lstm_units,
        Tau=Tau,
        C=C,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        lst_kernels=lst_kernels,
        lst_kernels_iaf=lst_kernels_iaf,
        f=f,
        k=k,
        batch_norm_moment=.99,
        l2_loss_eta=l2_loss_eta,
        dropout_rate=dropout_rate,
        check_error_x_recon=check_error_x_recon,
        check_error_z=check_error_z
    )
