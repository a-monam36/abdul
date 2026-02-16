# %%


import pandas as pd
import yfinance as yf
import pandas_ta_classic as pandas_ta
import numpy as np
import streamlit as st

import pandas_datareader.data as web
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from sklearn.cluster import KMeans



@st.cache_data
def get_sp500_data():

  sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', storage_options={'User-Agent': 'Mozilla/5.0'})[0]
  sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

  symbols_list = sp500['Symbol'].unique().tolist()

  end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
  start_date = pd.to_datetime(end_date) - pd.DateOffset(years=8)


  factor_data = yf.download(tickers= symbols_list, start=start_date, end=end_date, auto_adjust=False).stack()

  factor_data.index.names = ['date', 'ticker']

  factor_data.columns = factor_data.columns.str.lower()
  
  return factor_data




#(STEP 2)

def calculate_metrics(factor_data, use_rsi, use_gk, use_bb, use_atr, use_macd):


  #garman klass vol
  if use_gk:
    factor_data['garman_klass_vol'] = ((np.log(factor_data['high']) - np.log(factor_data['low']))**2) /2 - (2*np.log(2) - 1)*((np.log(factor_data['adj close']) - np.log(factor_data['open']))**2)

  # rsi for 20 days
  if use_rsi:
    factor_data['rsi'] = factor_data.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

  # bollinger bands for 20 days
  if use_bb:
    def compute_bbands(x):
      bands = pandas_ta.bbands(close=np.log1p(x), length=20)

      return bands.iloc[:, 0:3] # calculate high mid and low 

    bbands_factor_data = factor_data.groupby(level=1, group_keys=False)['adj close'].apply(compute_bbands)

    bbands_factor_data.columns = ['bb_low', 'bb_mid', 'bb_high']

    factor_data = factor_data.join(bbands_factor_data)

  # ATR for 14 days
  if use_atr:
    def compute_atr(stock_data):
      high = stock_data['high'].astype(float)
      low = stock_data['low'].astype(float)
      close = stock_data['close'].astype(float)

      if high.isnull().all() or low.isnull().all() or close.isnull().all():
        return pd.Series(np.nan, index=stock_data.index)

      atr = pandas_ta.atr(high= high, low=low, close= close, length=14)

      if atr.isnull().all():
        return pd.Series(np.nan, index=stock_data.index)

      return atr.sub(atr.mean()).div(atr.std()) # (x- mean)/ sd = z 

    factor_data['atr'] = factor_data.groupby(level=1, group_keys=False).apply(compute_atr)

  #macd for 20 days
  if use_macd:
    def compute_macd(close):
        macd = pandas_ta.macd(close=close, length=20).iloc[:,0] # ignore histogram and signal just use macd
        return macd.sub(macd.mean()).div(macd.std())

    factor_data['macd'] = factor_data.groupby(level=1, group_keys= False)['adj close'].apply(compute_macd)

  #dollar volume per million
  factor_data['dollar_volume'] = (factor_data['adj close']*factor_data['volume'])/1e6

  factor_data = factor_data.dropna()

  return factor_data


#(STEP 3)

def top_150_stocks(factor_data):


  last_cols = [c for c in factor_data.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                            'high', 'low', 'close']]

  data = (pd.concat([factor_data.unstack('ticker')['dollar_volume'].resample('ME').mean().stack('ticker').to_frame('dollar_volume'),
                    factor_data.unstack()[last_cols].resample('ME').last().stack('ticker')],
                    axis=1)).dropna() # make new columns that hold end of month data

  #calculating rolling average for 5 years

  data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

  #rank by date and value

  data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

  #drop 
  data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

  return data


#(STEP 4 )

def momentum(data):
  def calculate_returns(factor_data):

      outlier_cutoff = 0.005

      lags = [1, 2, 3, 6, 9, 12]

      for lag in lags:

          factor_data[f'return_{lag}m'] = (factor_data['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                      upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
      return factor_data
  

  data = data.groupby(level=1, group_keys = False).apply(calculate_returns).dropna()
  

  return data


#(STEP 5)

@st.cache_data
def get_fama_french_factors(start_date='2010-01-01'):
    ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start_date)[0].drop('RF', axis=1)
    ff_data.index = ff_data.index.to_timestamp()
    ff_data = ff_data.resample('ME').last().div(100)
    ff_data.index.name = 'date'
    return ff_data

def calculate_rolling_betas(data, factor_data):

    joint_df = factor_data.join(data['return_1m']).dropna()
    

    num_regressors = 6 


    observations = joint_df.groupby(level=1).size()
    
    valid_stocks = observations[observations > num_regressors]
    joint_df = joint_df[joint_df.index.get_level_values('ticker').isin(valid_stocks.index)]

    
    def run_rolling_ols(x):
        
        actual_nobs = x.shape[0]
        
        current_window = min(24, actual_nobs)
        
       
        if current_window <= num_regressors:
            return pd.DataFrame(np.nan, index=x.index, columns=x.drop('return_1m', axis=1).columns)

        return RollingOLS(
            endog=x['return_1m'], 
            exog=sm.add_constant(x.drop('return_1m', axis=1)),
            window=current_window,
            min_nobs=num_regressors + 1 
        ).fit(params_only=True).params.drop('const', axis=1)

    betas = joint_df.groupby(level=1, group_keys=False).apply(run_rolling_ols)
    
    
    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    data = data.join(betas.groupby('ticker').shift())
    data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))
    
    return data.dropna()











# --- THE EXECUTION PART ---

if __name__ == "__main__":

    raw_data = get_sp500_data()

    featured_data = calculate_metrics(
        raw_data, 
        use_rsi=True, 
        use_gk=True, 
        use_bb=True, 
        use_atr=True, 
        use_macd=True
    )

    filtered_factor_data = top_150_stocks(featured_data)
    final_result = momentum(filtered_factor_data)
    
    
    ff_factors = get_fama_french_factors()
    final_result = calculate_rolling_betas(final_result, ff_factors)
    
    print(final_result.tail())





# %%
