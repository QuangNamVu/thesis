import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

A_LIMIT = Decimal('.01')
B_LIMIT = Decimal('.0001')


def buy():
    global PRICE, A_FUND_AMOUNT, B_FUND_AMOUNT, fee_real
    # buy A sell B:
#     print("buy with price {}".format(PRICE))
    sell_amount_B = B_FUND_AMOUNT - (B_FUND_AMOUNT % B_LIMIT)

    cost = (sell_amount_B * fee_real) / PRICE

    B_FUND_AMOUNT -= sell_amount_B
    A_FUND_AMOUNT += (sell_amount_B / PRICE - cost)


def sell():
    global PRICE, A_FUND_AMOUNT, B_FUND_AMOUNT, fee_real
    # buy B sell A:
#     print("sell with price {}".format(PRICE))
    sell_amount_A = A_FUND_AMOUNT - (A_FUND_AMOUNT % A_LIMIT)

    cost = sell_amount_A * fee_real * PRICE

    A_FUND_AMOUNT -= sell_amount_A
    B_FUND_AMOUNT += (sell_amount_A * PRICE - cost)


test_file = '/home/nam/data/ccxt/BTC_USDT_binance_1h.csv'

df = pd.read_csv(test_file)
Close_lst = df.Close.values

fee_assume = Decimal('0.10')
fee_real = Decimal('0.001')

A_FUND_AMOUNT = Decimal('1.0')
B_FUND_AMOUNT = Decimal('0.0')
A_FUND_AMOUNT_LIST = []
B_FUND_AMOUNT_LIST = []
PRICE_LIST = []
PRICE = Decimal(str(Close_lst[0]))
PREV_SIDE = 'buy'
for NEW_PRICE in Close_lst[1:]:
    NEW_PRICE = Decimal(str(NEW_PRICE))
    if NEW_PRICE > PRICE / (Decimal('1.0') - fee_assume) and PREV_SIDE == 'buy':
        PRICE = NEW_PRICE
        sell()

        PREV_SIDE = 'sell'
        PRICE_LIST.append(PRICE)
        A_FUND_AMOUNT_LIST.append(A_FUND_AMOUNT)
        B_FUND_AMOUNT_LIST.append(B_FUND_AMOUNT)
    elif NEW_PRICE < PRICE * (Decimal('1.0') - fee_assume) and PREV_SIDE == 'sell':
        PRICE = NEW_PRICE
        buy()

        PREV_SIDE = 'buy'
        PRICE_LIST.append(PRICE)
        A_FUND_AMOUNT_LIST.append(A_FUND_AMOUNT)
        B_FUND_AMOUNT_LIST.append(B_FUND_AMOUNT)


print("Number of trade: ", len(A_FUND_AMOUNT_LIST))
print(A_FUND_AMOUNT, "  BTC")
print(B_FUND_AMOUNT, "  USDT")
print("Previous amount USDT is ", Decimal('1.0') *
      Decimal(Close_lst[0]) + B_FUND_AMOUNT, "  USDT")
print("Current amount USDT is ", A_FUND_AMOUNT *
      Decimal(Close_lst[-1]) + B_FUND_AMOUNT, "  USDT")
