# coding=utf-8

import ccxt
import os

# hitbtc   = ccxt.hitbtc({'verbose': True})
# bitmex   = ccxt.bitmex()
# huobipro = ccxt.huobipro()

binance_key = open("/home/nam/.ssh/binance", "r")
binance_key = binance_key.read().split('\n')
# binance     = ccxt.binance({
#     'apiKey': '7igelhtEw3GAITSkVv4pKXds2UDiMUdyZaTxA6EoEgAQeq8JO9g9nLDQOGXq2iIN',
#     'secret': binance_key,
# })

exchange_id = 'binance'
# exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': '7igelhtEw3GAITSkVv4pKXds2UDiMUdyZaTxA6EoEgAQeq8JO9g9nLDQOGXq2iIN',
    'secret': binance_key,
    'timeout': 30000,
    'enableRateLimit': True,
})

binance_markets = exchange.load_markets()

# print(exchange.id, hitbtc_markets)
print(exchange.id, binance_markets)
# print(exchange.id, hitbtc_markets)
print(exchange.id, binance_markets)
# print(exchange.id, exchange.load_markets())

# print(hitbtc.fetch_order_book(hitbtc.symbols[0]))
# print(bitmex.fetch_ticker('BTC/USD'))
# print(huobipro.fetch_trades('LTC/CNY'))

# print(exmo.fetch_balance())

# # sell one ฿ for market price and receive $ right now
# print(exmo.id, exmo.create_market_sell_order('BTC/USD', 1))

# # limit buy BTC/EUR, you pay €2500 and receive ฿1  when the order is closed
# print(exmo.id, exmo.create_limit_buy_order('BTC/EUR', 1, 2500.00))

# # pass/redefine custom exchange-specific order params: type, amount, price, flags, etc...
# kraken.create_market_buy_order('BTC/USD', 1, {'trading_agreement': 'agree'})
