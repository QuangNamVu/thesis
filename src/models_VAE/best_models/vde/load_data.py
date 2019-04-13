import numpy as np
import pandas as pd
from tensorpack.dataflow.base import RNGDataFlow


def load_data_seq(hps):
    # df = resample_data(info_params)
    df = pd.read_csv(filepath_or_buffer=hps.data_file_name)
    lag_time = hps.lag_time
    # remove nan at end of array
    next_delta = np.roll(df['Close'].diff(periods=1).values, shift=-1)[0:-1]

    if hps.C is 3:
        # return -1 0 1 => 0 1 2 for classify
        next_movement = np.sign(next_delta).astype(int) + 1
    else:
        next_movement = np.where(next_delta >= 0, 1, 0)

    # lag_time + 1 + 1 -> next offset [0: 10] = [129:139] -> 131 in csv
    hidden_target_classify = np.roll(next_movement, - lag_time, axis=0)  # shift up

    # one-hot encode with numpy
    target_one_hot = np.eye(hps.C)[np.reshape(hidden_target_classify, -1)]

    attributes_normalize_mean = hps.attributes_normalize_mean
    attributes_normalize_log = hps.attributes_normalize_log

    n_attributes_total = attributes_normalize_mean + attributes_normalize_log

    X_selected = df[n_attributes_total]
    X_diff = X_selected.diff(periods=hps.lag_time).dropna
    X_0 = X_diff().copy()

    for att in attributes_normalize_log:
        X_0[att] = np.log(X_diff[att] + 1e-20)

    idx_split = hps.N_train_seq
    n_test_seq = hps.N_test_seq
    T, M = hps.T, hps.M

    #
    if hps.normalize_data_idx:
        mu = np.mean(X_0[0:idx_split])
        X_max = np.max(X_0[0:idx_split])
        X_min = np.min(X_0[0:idx_split])
        X_std = np.std(X_0[0:idx_split])

    else:
        mu = np.mean(X_0)
        X_max = np.max(X_0)
        X_min = np.min(X_0)
        X_std = np.std(X_0)

    normalize = hps.normalize_data
    if normalize is 'z_score':
        print('Normalize: Z score')
        X_normalized = (X_0 - mu) / X_std

    elif normalize is 'min_max':
        print('Normalize: Min Max')
        X_normalized = (X_0 - X_min) / (X_max - X_min)

    elif normalize is 'min_max_centralize':
        print('Normalize: Min Max Centralize')
        X_normalized = (X_0 - mu) / (X_max - X_min)

    else:
        print('Missing Normalization')
        X_normalized = X_0

    X_normalized = X_normalized.values

    tau = hps.Tau

    X_tau_seq = np.roll(X_normalized, - lag_time, axis=0)

    X_train_seq = X_normalized[0:idx_split]
    X_test_seq = X_normalized[idx_split:idx_split + n_test_seq]

    X_tau_train_seq = X_tau_seq[0:idx_split]
    X_tau_test_seq = X_tau_seq[idx_split:idx_split + n_test_seq]

    if not hps.create_next_trend:
        return X_train_seq, X_test_seq, X_tau_train_seq, X_tau_test_seq

    idx_split_predict = idx_split + tau - T
    target_one_hot_train_seq = target_one_hot[0:idx_split_predict]
    target_one_hot_test_seq = target_one_hot[idx_split:idx_split_predict + n_test_seq]
    return X_train_seq, X_test_seq, X_tau_train_seq, X_tau_test_seq, target_one_hot_train_seq, target_one_hot_test_seq


def segment_seq(X_seq, X_tau_seq, hps, target_one_hot=None):
    # assert info_params.n_time_steps * info_params.n_batch > 0, 'Batch size = 0 num batch = 0'

    assert len(np.shape(X_seq)) >= 2, 'Dim Tensor >= 3'

    T, D = hps.T, hps.D
    Tau = hps.Tau
    N = X_seq.shape[0]  # n_samples N = T -> 1 segment; N = T + 1 -> 2 segments

    # window slide
    n_segments = N - T + 1
    X_segment = np.zeros(shape=(n_segments, T, D))
    X_tau_segment = np.zeros(shape=(n_segments, T, D))

    for i in range(n_segments):
        X_segment[i] = X_seq[i: i + T]
        X_tau_segment[i] = X_tau_seq[i: i + T]

    if not hps.create_next_trend:
        return X_segment, X_tau_segment

    assert type(target_one_hot) is np.ndarray, 'Trending classify is missing'

    y_one_hot_segment = np.zeros(shape=(n_segments, Tau, hps.C))

    for i in range(n_segments):
        y_one_hot_segment[i] = target_one_hot[i: i + Tau]

    return X_segment, X_tau_segment, y_one_hot_segment


class LoadData(RNGDataFlow):

    def __init__(self, X, X_tau, y_one_hot, shuffle=False):

        self.shuffle = shuffle

        self.x = X
        self.x_hat = X_tau
        self.y_one_hot = y_one_hot

    def __len__(self):
        return self.x.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            x_element = self.x[k]
            x_hat_element = self.x_hat[k]
            y_one_hot_element = self.y_one_hot[k]
            yield [x_element, x_hat_element, y_one_hot_element]
