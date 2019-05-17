import numpy as np
import os

home_path = os.path.expanduser("~") + '/'
from info_params import get_default_hparams
from utils.load_data import *
from sklearn.svm import SVC


def get_data(hps):
    dfX, df_next_deltaClose = load_data_seq(hps)

    segment, next_segment, target_one_hot = segment_seq(dfX, df_next_deltaClose, hps)

    train_segment, test_segment, _, _, train_target_one_hot, test_target_one_hot = \
        train_test_split(segment, next_segment, target_one_hot, hps)

    return train_segment, test_segment, train_target_one_hot, test_target_one_hot


hps = get_default_hparams()
hps.data_file_name = home_path + "data/cryptodatadownload/1-order.csv"
hps.attributes_normalize_mean = ['Close', 'Volume BTC']#, 'Spread High-Low', 'Spread Close-Open']
hps.is_concat = False
hps.Tau = 1
hps.T = 30
hps.C = 2
hps.D = (1 + hps.is_concat) * len(hps.attributes_normalize_mean)
hps.is_differencing = True
hps.lag_time = 1
hps.N_train_seq = 10000
hps.normalize_data = 'z_score'
hps.normalize_data_idx = True

X_train, X_test, train_target_one_hot, test_target_one_hot = get_data(hps)
X_train = np.reshape(X_train, newshape=[-1, hps.T * hps.D])
X_test = np.reshape(X_test, newshape=[-1, hps.T * hps.D])

y_train = np.argmax(train_target_one_hot, axis=-1)
y_train = np.reshape(y_train, newshape=[-1, hps.Tau])

y_test = np.argmax(test_target_one_hot, axis=-1)
y_test = np.reshape(y_test, newshape=[-1, hps.Tau])

svclassifier = SVC(C=1, kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pred)
rp = classification_report(y_test, y_pred)

print(cm)
print(rp)
