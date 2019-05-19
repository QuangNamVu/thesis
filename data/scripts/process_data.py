import numpy as np
import pandas as pd
from datetime import datetime
import os

home_path = os.path.expanduser("~") + '/'
data_file_name = home_path + 'data/cryptodatadownload/tac_CoinBase_BTC_1h.csv'
date_fmt = '%Y-%m-%d %I-%p'

df = pd.read_csv(filepath_or_buffer=data_file_name)

df['Date_Unix'] = df['Date'].apply(lambda x: datetime.strptime(x, date_fmt))
df['Date_Unix'] = pd.to_numeric(df['Date_Unix'])
df['Spread High-Low'] = df.High - df.Low
df['Spread Close-Open'] = df.Close - df.Open

lst_attributes = ['Open', 'High', 'Low', 'Close', 'Spread High-Low', 'Spread Close-Open', 'Volume BTC', 'Volume USD']

# delta_Open, delta_High, ...
for att in lst_attributes:
    df['delta_' + att] = df[att].diff(periods=1)

# Moving Average
w = 24
df["MA_Close_" + str(w)] = df.Close.rolling(window=w).mean()
df["MA_V_BTC_" + str(w)] = df["Volume BTC"].rolling(window=w).mean()

w = 240
df["MA_Close_" + str(w)] = df.Close.rolling(window=w).mean()
df["MA_V_BTC_" + str(w)] = df["Volume BTC"].rolling(window=w).mean()

df = df.dropna()
print(df.head())
df.round(6).to_csv(home_path + 'Dropbox/data/processed_data.csv', index=None, header=True)
