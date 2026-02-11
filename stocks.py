# %%


import pandas as pd
import yfinance as yf
import pandas_ta_classic as pandas_ta
import numpy as np






sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', storage_options={'User-Agent': 'Mozilla/5.0'})[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

symbols_list = sp500['Symbol'].unique().tolist()

end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
start_date = pd.to_datetime(end_date) - pd.DateOffset(years=8)


df = yf.download(tickers= symbols_list, start=start_date, end=end_date, auto_adjust=False).stack()

df.index.names = ['date', 'ticker']

df.columns = df.columns.str.lower()

# After the sp500 = pd.read_html(...) line:
print(f"Successfully loaded {len(sp500)} rows from the S&P 500 table.")




#STEP 2

df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0]) # selecting lower part from bb function

df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1]) # selecting mid part

df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2]) # selecting high

def compute_atr(stock_data): # for 14 days
  high = stock_data['high'].astype(float)
  low = stock_data['low'].astype(float)
  close = stock_data['close'].astype(float)

  if high.isnull().all() or low.isnull().all() or close.isnull().all():
    return pd.Series(np.nan, index=stock_data.index)

  atr = pandas_ta.atr(high= high, low=low, close= close, length=14)

  if atr.isnull().all():
    return pd.Series(np.nan, index=stock_data.index)

  return atr.sub(atr.mean()).div(atr.std())


df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys= False)['adj close'].apply(compute_macd)

df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

df.dropna()

