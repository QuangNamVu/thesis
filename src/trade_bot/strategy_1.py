import ccxt
import time
import os
import datetime


msec = 1000
minute = 60 * msec
hour = 60 * minute

symbol = 'BNB/BTC'
market = 'binance'
timewindow = '1m'

if timewindow == '1h':
    offset = hour
    delay = offset / 1000
elif timewindow == '1m':
    offset = minute
    delay = offset / 1000


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

# exchange = ccxt.binance({
#     # 'apiKey': binance.apiKey,
#     # 'secret': binance.secret,
#     'timeout': 30000,
#     'rateLimit': 2000,
#     'enableRateLimit': True
# })
exchange = binance

now = datetime.datetime.now()
from_timestamp = exchange.parse8601(now)


header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

PREV_SIDE = "buy"
# price in previous trade
# PREV_PRICE = -1
PREV_PRICE = 0.0021111
# CLOSE_PRICE = -1

# binance's fee trading is .075% for pair contain BNB and 1% for others
# https://www.binance.com/en/fee/schedule
# assume fee is 1%
fee = .01
BNB_LIMIT = .01
BTC_LIMIT = .0001


def check_and_trade():
    global exchange, PREV_SIDE, PREV_PRICE, CLOSE_PRICE
    global fee

    if PREV_SIDE is "buy":
        # if True:
        'now: sell when y_hat > y/(1 - fee)'
        if CLOSE_PRICE <= PREV_PRICE/(1.0 - fee):
            #print("Close price: {} Previous price: {}, still waiting for sell".format( CLOSE_PRICE, PREV_PRICE))
            print("Close price: {} Previous price: {}, still waiting for sell {:.4}%".format(
                CLOSE_PRICE, PREV_PRICE, (CLOSE_PRICE - PREV_PRICE)*100 / PREV_PRICE))
            return
        # amount = 1.0  # BNB
        # amount = max btc * price
        amount = exchange.fetch_balance().get('free').get('BNB')
        if amount < BNB_LIMIT:
            print("Not enough to sell")
            return
        price = CLOSE_PRICE
        print("------------------------------------------------------")
        print("SELLING BNB Close price: {} Previous price: {}".format(
            CLOSE_PRICE, PREV_PRICE))
        order = exchange.create_order(symbol, 'limit', 'sell', amount, price)

        PREV_SIDE = "sell"
        PREV_PRICE = CLOSE_PRICE

    elif PREV_SIDE is "sell":
        # if True:
        'now: buy when y_hat < y*(1 - fee)'
        if CLOSE_PRICE >= PREV_PRICE*(1.0 - fee):
            print("Close price: {} Previous price: {}, still waiting for buy {:.4}%".format(
                CLOSE_PRICE, PREV_PRICE, (CLOSE_PRICE - PREV_PRICE)*100 / PREV_PRICE))
            return
        # print("SELL PAIR")
        # amount = 1.0  # BNB
        amount = binance.fetch_balance().get('free').get('BTC')
        if amount < BTC_LIMIT:
            print("Not engouh to buy")
            return
        amount /= CLOSE_PRICE
        price = CLOSE_PRICE
        print("------------------------------------------------------")
        print("BUYING BNB Close price: {} Previous price: {}".format(
            CLOSE_PRICE, PREV_PRICE))
        order = binance.create_order(symbol, 'limit', 'buy', amount, price)
        print(order)

        PREV_SIDE = "buy"
        PREV_PRICE = CLOSE_PRICE


while True:
    try:
        ohlcvs = exchange.fetch_ohlcv(symbol, timewindow, from_timestamp)

        while len(ohlcvs) == 0:
            print("Waiting for incomming fetch")
            time.sleep(delay)
            ohlcvs = exchange.fetch_ohlcv(symbol, timewindow, from_timestamp)
            # if previous price is not inited and new fetch is pushed
            # if len(ohlcvs) != 0 and PREV_PRICE == -1: PREV_PRICE = ohlcvs[-1][4]

        if PREV_PRICE == -1:
            PREV_PRICE = ohlcvs[-1][4]

        print(exchange.milliseconds(), 'Fetched', len(ohlcvs), 'candles')
        if len(ohlcvs) > 0:
            first = ohlcvs[0][0]
            last = ohlcvs[-1][0]

            print('First candle epoch', first, exchange.iso8601(first))
            # previous price = last close price

            # if previous price is set: trade
            if PREV_PRICE != -1:
                CLOSE_PRICE = ohlcvs[-1][4]
                check_and_trade()

            from_timestamp = ohlcvs[-1][0] + offset
            # from_timestamp = ohlcvs[-1][0]

            # v = ohlcvs[0][0]/ 1000
            # !date --date @{v} +"%Y-%m-%d %H:%M"
            print('Last candle epoch', last, exchange.iso8601(last))

        now = datetime.datetime.now()
        to_datetime = '{:%Y-%m-%d %H:%M:%S}'.format(now)
        to_timestamp = exchange.parse8601(to_datetime)

    except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
        print('Got an error', type(error).__name__,
              error.args, ', retrying in', offset, 'seconds...')
        # time.sleep(delay)
