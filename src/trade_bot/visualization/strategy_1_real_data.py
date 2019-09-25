import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decimal import Decimal

# import data

# BNB fund amount
A_FUND_AMOUNT = Decimal('1.18')
# BTC fund amount
B_FUND_AMOUNT = Decimal('0.0')

PRICE = Decimal('0.0021111')
# PRICE = 0.0021111

# CLOSE_PRICE = -1
# binance's fee trading is .075% for pair contain BNB and 1% for others
# https://www.binance.com/en/fee/schedule

# assume fee is 1%
fee_assume = Decimal('.01')
# fee in real:
fee_real = Decimal('.00075')

A_LIMIT = Decimal('.01')
B_LIMIT = Decimal('.0001')


def buy():
    global PRICE, A_FUND_AMOUNT, B_FUND_AMOUNT
    # buy A sell B:
    sell_amount_B = B_FUND_AMOUNT - (B_FUND_AMOUNT % B_LIMIT)

    cost = (sell_amount_B * fee_real) / PRICE

    B_FUND_AMOUNT -= sell_amount_B
    A_FUND_AMOUNT += (sell_amount_B / PRICE - cost)


def sell():
    global PRICE, A_FUND_AMOUNT, B_FUND_AMOUNT
    # buy B sell A:
    sell_amount_A = A_FUND_AMOUNT - (A_FUND_AMOUNT % A_LIMIT)

    cost = sell_amount_A * fee_real * PRICE

    A_FUND_AMOUNT -= sell_amount_A
    B_FUND_AMOUNT += (sell_amount_A * PRICE - cost)


A_FUND_AMOUNT_LIST = []
B_FUND_AMOUNT_LIST = []
PRICE_LIST = []

for i in range(N):
    PRICE = PRICE / (1 - fee_assume)
    PRICE_LIST.append(PRICE)
    sell()
    A_FUND_AMOUNT_LIST.append(A_FUND_AMOUNT)
    B_FUND_AMOUNT_LIST.append(B_FUND_AMOUNT)

    PRICE = PRICE * (1 - fee_assume)
    PRICE_LIST.append(PRICE)
    buy()
    A_FUND_AMOUNT_LIST.append(A_FUND_AMOUNT)
    B_FUND_AMOUNT_LIST.append(B_FUND_AMOUNT)


# print(A_FUND_AMOUNT_LIST)
# print(B_FUND_AMOUNT_LIST)
# print(PRICE_LIST)

A_FUND_AMOUNT_LIST = np.array(A_FUND_AMOUNT_LIST)
fig, axs = plt.subplots(3)
fig.suptitle('Threshold 1% with fee .075%')
axs[0].plot(np.arange(2 * N), A_FUND_AMOUNT_LIST, label='A Amount')
axs[1].plot(np.arange(2 * N), B_FUND_AMOUNT_LIST, label='B Amount')
axs[2].plot(np.arange(2 * N), PRICE_LIST, label='Price')

plt.legend()
plt.show()
fig.savefig('strategy_01_fee_assume_.01.png')
