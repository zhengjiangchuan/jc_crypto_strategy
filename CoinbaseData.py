from coinbase.rest import RESTClient
from datetime import datetime
import os
import pandas as pd
from datetime import timedelta

def get_api_keys(is_alternative = False):
    crypto_folder = os.getenv("CRYPTO")
    print("crypto_folder")
    print(crypto_folder)

    #key_file = "jc_alternative.txt" if is_alternative else "jc.txt"

    key_file = "jc_alternative.txt" if not is_alternative else "jc.txt"

    fd = open(os.path.join(crypto_folder, key_file))
    lines = []
    for line in fd:
        lines += [line[:-1]]

    api_key = lines[0]
    api_secret = lines[1]
    return (api_key, api_secret)

api_key, api_secret = get_api_keys(is_alternative=True)
client = RESTClient(api_key=api_key,
                            api_secret=api_secret)


end_time = datetime(2024,11,15,0,0,0)
start_time = end_time - timedelta(hours=300)

start_time = int(start_time.timestamp())
end_time = int(end_time.timestamp())

response = client.get_candles(product_id='BONK-USDC', start=start_time, end=end_time, granularity='ONE_HOUR')

candles = response['candles']

columns = ['datetime', 'open', 'high', 'low', 'close']
data = []

for candle in candles:
    data += [[datetime.fromtimestamp(int(candle['start'])), float(candle['open']), float(candle['high']),
              float(candle['low']), float(candle['close'])]]

data_df = pd.DataFrame(data=data, columns=columns)

data_df['open'] = data_df['open'] * 10000
data_df['high'] = data_df['high'] * 10000

print(data_df.iloc[0:30])

print(data_df.iloc[-20:])