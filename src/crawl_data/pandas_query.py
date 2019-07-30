import pandas as pd
from pymongo import MongoClient

mongo_client = MongoClient('localhost', 27017)
db = mongo_client.crypto_currency
collection = db['ohlcv']
symbol = 'BNB/BTC'
market = 'binance'
timewindow = '1h'

def read_mongo(collection, query={}, no_id=True):
    """ Read from Mongo and Store into DataFrame """

    # Make a query to the specific DB and Collection
    cursor = collection.find(query)

    # Expand the cursor and construct the DataFrame
    df =  pd.DataFrame(list(cursor))

    # Delete the _id
    if no_id:
        del df['_id']
    return df


query = {
        "symbol": {"$eq": symbol},
        "market": {"$eq": market},
        "timewindow": {"$eq": timewindow},
        }

df = read_mongo(collection,query= query)
print(df.shape)