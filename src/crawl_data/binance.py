import ccxt
import time
import os
import pandas as pd
import datetime
from pymongo import MongoClient

mongo_client = MongoClient('localhost', 27017)
db = mongo_client.crypto_currency
collection = db['ohlcv']
symbol = 'BNB/BTC'
market = 'binance'
timewindow = '1h'


def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)


binance = ccxt.binance()
home_path = os.path.expanduser("~")
binance.apiKey = get_file_contents(home_path + '/api_key/binance/pub')
binance.secret = get_file_contents(home_path + '/api_key/binance/private')

# upon instantiation
binance = ccxt.binance({
    'apiKey': binance.apiKey,
    'secret': binance.secret,
})

exchange = ccxt.binance({
    # 'apiKey': binance.apiKey,
    # 'secret': binance.secret,
    'timeout': 30000,
    'rateLimit': 2000,
    'enableRateLimit': True
})


msec = 1000
minute = 60 * msec
hour = 60 * minute
hold = hour + 5 * minute

from_datetime = '2019-03-28 00:00:00'
from_timestamp = exchange.parse8601(from_datetime)


now = datetime.datetime.now()
to_datetime = '{:%Y-%m-%d %H:%M:%S}'.format(now)
to_timestamp = exchange.parse8601(to_datetime)

# now = exchange.milliseconds()

header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']


while from_timestamp < to_timestamp:
    try:
        # print(exchange.milliseconds(), 'Fetching candles starting from', exchange.iso8601(from_timestamp))
        
        ohlcvs = exchange.fetch_ohlcv(symbol, timewindow, from_timestamp)

        if len(ohlcvs) == 0:
            print("waiting for incomming fetch")
            time.sleep(hold)
            ohlcvs = exchange.fetch_ohlcv(symbol, timewindow, from_timestamp)

        # df_current = pd.DataFrame(list(ohlcvs), columns = header)
        df_current = pd.DataFrame(ohlcvs, columns = header)
        df_current['market'] = market
        df_current['symbol'] = symbol
        df_current['timewindow'] = timewindow
        # convert df to list of dict
        lst_dict = df_current.T.to_dict().values()

        collection.insert_many(lst_dict)

        print(exchange.milliseconds(), 'Fetched', len(ohlcvs), 'candles')
        if len(ohlcvs) > 0:
            first = ohlcvs[0][0]
            last = ohlcvs[-1][0]
            print('First candle epoch', first, exchange.iso8601(first))

            # from_timestamp += len(ohlcvs) * minute * 5  # very bad
            from_timestamp = ohlcvs[-1][0] + minute * 5  # good
            # from_timestamp = ohlcvs[-1][0] 

            # v = ohlcvs[0][0]/ 1000
            # !date --date @{v} +"%Y-%m-%d %H:%M"
            print('Last candle epoch', last, exchange.iso8601(last))

        now = datetime.datetime.now()
        to_datetime = '{:%Y-%m-%d %H:%M:%S}'.format(now)
        to_timestamp = exchange.parse8601(to_datetime)

    except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
        print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
        time.sleep(hold)


# dumping

# !ipython
# i = 0
# v = ohlcvs[i][0] / 1000
# !date --date @{v} +"%Y-%m-%d %H:%M"
# ohlcvs[i][1] # High