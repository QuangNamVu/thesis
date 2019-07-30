import pandas
from pymongo import MongoClient

mongo_client = MongoClient('localhost', 27017)
db = mongo_client.crypto_currency
collection = db['ohlcv']
symbol = 'BNB/BTC'
market = 'binance'
timewindow = '1h'