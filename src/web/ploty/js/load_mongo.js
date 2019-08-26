var MongoClient = require('mongodb').MongoClient;

function mongodb_query(url, db_name, col_name, query_str) {
    MongoClient.connect(url, { useNewUrlParser: true }, function(err, db) {
        if (err) throw err;
        var dbo = db.db(db_name);
        dbo.collection(col_name).find(query_str).toArray(function(err, rs) {
            if (err) throw err;
            db.close();
            console.log(rs);
            // return rs;
        });
    });
}

var url = "mongodb://localhost:27017/";
var query_str = { "market": "binance", "symbol": "BNB/BTC", "timewindow": "1h" };
db_name = "crypto_currency";
col_name = 'ohlcv';

var query_val = mongodb_query(url, db_name, col_name, query_str);

console.log(query_val)