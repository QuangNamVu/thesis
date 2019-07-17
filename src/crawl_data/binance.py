import ccxt

def get_file_contents(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)


binance = ccxt.binance()
binance.apiKey = get_file_contents('/home/nam/api_key/binance/pub')
binance.secret = get_file_contents('/home/nam/api_key/binance/private')

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

symbols = ['BTC/USDT']

import time
msec = 1000
minute = 60 * msec
hour = 60 * minute
hold = 30

from_datetime = '2019-03-18 00:00:00'
from_timestamp = exchange.parse8601(from_datetime)

to_datetime = '2019-03-20 00:00:00'
to_timestamp = exchange.parse8601(to_datetime)

# now = exchange.milliseconds()

data = []

# while from_timestamp < to_timestamp:
while from_timestamp < 1000000009999000:

    try:
        # print(exchange.milliseconds(), 'Fetching candles starting from', exchange.iso8601(from_timestamp))
        ohlcvs = exchange.fetch_ohlcv('BTC/USDT', '1m', from_timestamp)
        print(exchange.milliseconds(), 'Fetched', len(ohlcvs), 'candles')
        if len(ohlcvs) > 0:
            first = ohlcvs[0][0]
            last = ohlcvs[-1][0]
            print('First candle epoch', first, exchange.iso8601(first))

            # from_timestamp += len(ohlcvs) * minute * 5  # very bad
            # from_timestamp = ohlcvs[-1][0] + minute * 5  # good
            from_timestamp = ohlcvs[-1][0] + 2 * hour
            data += ohlcvs
            print(ohlcvs[-1])

            # v = ohlcvs[0][0]/ 1000
            # !date --date @{v} +"%Y-%m-%d %H:%M"

            print('Last candle epoch', last, exchange.iso8601(last))

    except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
        print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
        time.sleep(hold)

# !ipython
# i = 0
# v = ohlcvs[i][0] / 1000
# !date --date @{v} +"%Y-%m-%d %H:%M"
# ohlcvs[i][1] # High