import os
import math
from tf_utils.hparams import *

home_path = os.path.expanduser("~") + '/'

data_file_name = home_path + 'data/cryptodatadownload/Coinbase_tiny.csv'

attributes_normalize_mean = ['Open', 'High', 'Low', 'Close', 'Volume BTC', 'Volume USD']

# attributes_normalize_mean = ['DeltaOpen1', 'DeltaHigh1', 'DeltaLow1', 'DeltaClose1', 'DeltaVolume_BTC1',
#                              'Delta_Volume_USD1', 'DeltaOpen200', 'DeltaHigh200', 'DeltaLow200', 'DeltaClose200',
#                              'DeltaVolume_BTC200', 'Delta_Volume_USD200']

attributes_normalize_log = []
D = len(attributes_normalize_log + attributes_normalize_mean)
is_VAE = True
is_VDE = True
is_IAF = False
# normalize_data = 'z_score'
normalize_data = 'min_max'
# 'z_score' 'min_max' 'default' 'min_max_centralize'

T = 30
lag_time = 10  # lag time represent recon should be T
Tau = 1
k = math.sqrt(T / Tau)
k = 5

# f = np.array(np.array([600, 150, 2]) * T * D / 100).astype(int)

f = [32, 8]
lst_kernels = [3, 3]
# n_z = int(T / k)
n_z = 10

normalize_data_idx = True
# normalize_data_idx = False
row_count = sum(1 for row in data_file_name)
N_train_seq = 5000
file = open(data_file_name)
numline = len(file.readlines())
N_test_seq = numline - N_train_seq - T - 2
check_error_x_recon = True
check_error_z = True

M = 100  # Batch size
l2_loss = 1e-6

lstm_units = 128
create_next_trend = True
C = 3
steps_per_epoch = 10
epochs = 20000

lst_kernels_iaf = [3, 3]

batch_norm_moment = .99

dropout_rate = .5


# self.idx_split = 262144 // self.check_per_N_mins  # Split sequence to test and train
# self.n_test_seq = 131072 // self.check_per_N_mins


def get_default_hparams():
    return HParams(

        data_file_name=data_file_name,

        normalize_data=normalize_data,
        is_VAE=is_VAE,
        is_VDE=is_VDE,
        is_IAF=is_IAF,
        normalize_data_idx=normalize_data_idx,

        attributes_normalize_mean=attributes_normalize_mean,
        attributes_normalize_log=attributes_normalize_log,
        T=T,
        lag_time=lag_time,
        M=M,
        N_train_seq=N_train_seq,
        N_test_seq=N_test_seq,

        learning_rate=1e-3,

        l2_loss=l2_loss,  # l2 regularization
        d_inputs=D,
        d_outputs=D,
        D=D,
        n_z=n_z,
        lstm_units=lstm_units,
        create_next_trend=create_next_trend,  # True if predict next seq
        Tau=Tau,
        C=C,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        lst_kernels=lst_kernels,
        lst_kernels_iaf=lst_kernels_iaf,
        f=f,
        batch_norm_moment=.99,
        dropout_rate=dropout_rate,
        check_error_x_recon=check_error_x_recon,
        check_error_z=check_error_z
    )

# self.data_file_name = home_path + 'data/cryptodatadownload/Coinbase_BTCUSD_1h.csv'
# self.data_file_name = home_path + 'data/yahoo_finance/processed_data.csv'
# data_file_name=home_path + 'data/cryptodatadownload/processed_data.csv',
# self.normalize_data = 'min_max_centralize'

# self.data_file_name = home_path + 'data/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv'
# data_file_name = home_path + 'data/cryptodatadownload/processed_data.csv'



# attributes_normalize_mean = ['DeltaOpen', 'DeltaHigh', 'DeltaLow', 'DeltaClose', 'DeltaVolume_BTC', 'Delta_Volume_USD']
# attributes_normalize_mean = ['DeltaOpen200', 'DeltaHigh200', 'DeltaLow200', 'DeltaClose200', 'DeltaVolume_BTC200',
#                              'Delta_Volume_USD200']
