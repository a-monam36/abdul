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
import matplotlib.pyplot as plt
import datetime as dt
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler



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
    def compute_rsi(x):
      rsi = pandas_ta.rsi(close=x, length=20)
      return rsi.sub(rsi.mean()).div(rsi.std())
    factor_data['rsi'] = factor_data.groupby(level=1)['adj close'].transform(compute_rsi)

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
  def calculate_returns(group_df):

      outlier_cutoff = 0.005

      lags = [1, 2, 3, 6, 9, 12]

      for lag in lags:

          group_df[f'return_{lag}m'] = (group_df['adj close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                      upper=x.quantile(1-outlier_cutoff)))
                                .add(1)
                                .pow(1/lag)
                                .sub(1))
      return group_df
  

  data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()
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
    data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.ffill().bfill())
    
    return data.dropna()


#(STEP 6 kmeans clustering )

initial_centroids = np.array([
    # Cluster 0: "The Rocket Ships" (High Momentum + High Profitability)
    # Target: Best for Growth Investors
    [1.5, 1.2, 0.5, -0.5, 1.0, 0.0, 1.2], 

    # Cluster 1: "The Deep Value Play" (Low RSI + High HML)
    # Target: Best for Contrarian/Value Investors
    [-0.5, 0.8, 0.2, 1.5, 0.2, 0.5, -1.5],

    # Cluster 2: "The Defensive Giants" (Low Beta + High Quality)
    # Target: Best for Conservative Investors (Safe Havens)
    [0.2, 0.5, -1.0, 0.2, 1.2, 1.0, 0.0],

    # Cluster 3: "The Underperformers" (Negative Momentum + High Volatility)
    # Target: Stocks to AVOID
    [-1.5, 1.5, 0.8, -0.2, -1.0, -0.5, -0.2]
])

def calculate_clusters(data):
   
  data = data.drop('cluster', axis=1, errors='ignore')
  cluster_cols = ['return_1m', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'rsi']
  def get_clusters(df):
      scaler = StandardScaler()
      df[cluster_cols] = scaler.fit_transform(df[cluster_cols])
      df['cluster'] = KMeans(n_clusters=4,
                            random_state=0,
                            init=initial_centroids).fit(df[cluster_cols]).labels_
      return df

  data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)
  return data



plt.style.use('ggplot')

def plot_all_clusters(data):
    # This list will hold all the 'plates' (charts) we create
    figures = []
    
    # --- YOUR LOOP LOGIC ---
    # get_level_values('date') pulls all the dates from your MultiIndex
    for i in data.index.get_level_values('date').unique().tolist():
        
        # 'g' is a 'Cross-Section' (xs) - it takes just the stocks for date 'i'
        g = data.xs(i, level=0)
        
        # Create a new figure for each date
        fig = plt.figure(figsize=(10, 6))
        plt.title(f'Cluster Map: {i.strftime("%Y-%m-%d")}')
        
        # Call your existing logic
        cluster_0 = g[g['cluster']==0]
        cluster_1 = g[g['cluster']==1]
        cluster_2 = g[g['cluster']==2]
        cluster_3 = g[g['cluster']==3]

        plt.scatter(cluster_0.iloc[:,0], cluster_0.iloc[:,6], color='red', label='Cluster 0')
        plt.scatter(cluster_1.iloc[:,0], cluster_1.iloc[:,6], color='green', label='Cluster 1')
        plt.scatter(cluster_2.iloc[:,0], cluster_2.iloc[:,6], color='blue', label='Cluster 2')
        plt.scatter(cluster_3.iloc[:,0], cluster_3.iloc[:,6], color='black', label='Cluster 3')
        
        plt.legend()
        plt.xlabel("Returns (1m)")
        plt.ylabel("RSI (20d)")
        
        # Instead of plt.show(), save it to our list
        figures.append(fig)
        
    return figures


def select_stocks(data):
    # 1. Filter for Cluster 0
    filtered_df = data[data['cluster'] == 0].copy()
    
    # 2. Reset ticker so it's a column, but keep date as the index
    filtered_df = filtered_df.reset_index(level=1)

    # 3. Shift the date by 1 day (to avoid look-ahead bias)
    # We apply this to the index itself
    filtered_df.index = filtered_df.index + pd.DateOffset(days=1)

    # 4. Re-establish the MultiIndex (date, ticker)
    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

    # 5. Get unique dates to iterate through
    dates = filtered_df.index.get_level_values('date').unique().tolist()

    fixed_dates = {}

    for d in dates:
        # Extract tickers for this specific date
        fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

    return fixed_dates



def portfolio_optimization(data, fixed_dates, max_weight=0.1, lookback=12):

    def optimize_weights(prices, lower_bound=0):
        returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
        covariance = risk_models.sample_cov(prices=prices, frequency=252)
        
        # solver='SCS' is more robust for constrained problems
        ef = EfficientFrontier(expected_returns=returns, cov_matrix=covariance, weight_bounds=(lower_bound, max_weight), solver='SCS')
        
        weights = ef.max_sharpe()
        return ef.clean_weights()

    # 1. Prepare Tickers and Date Range
    stocks = data.index.get_level_values('ticker').unique().tolist()
    start = data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=lookback)
    end = data.index.get_level_values('date').unique()[-1]
    
    # 2. Download Data
    new_df = yf.download(tickers=stocks, start=start, end=end, auto_adjust=False)

    # --- FIX 1: Handle MultiIndex Lowercasing safely ---
    if isinstance(new_df.columns, pd.MultiIndex):
        new_df.columns = new_df.columns.set_levels(new_df.columns.levels[0].str.lower(), level=0)
    else:
        new_df.columns = new_df.columns.str.lower()
    
    # Define which price column to use (usually 'adj close')
    price_col = 'adj close' if 'adj close' in new_df.columns.get_level_values(0) else 'close'

    # --- FIX 2: Calculate returns_df before the loop ---
    # This is needed for the performance calculation later
    returns_df = np.log(new_df[price_col]).diff()
    
    portfolio_df = pd.DataFrame()

    for start_date in fixed_dates.keys():
        try:
            # Setup current trading month and look-back period
            end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            cols = fixed_dates[start_date]
            
            optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=lookback)).strftime('%Y-%m-%d')
            optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            
            # --- FIX 3: Use the lowercase variable and valid columns check ---
            available_tickers = new_df[price_col].columns.tolist()
            valid_cols = [ticker for ticker in cols if ticker in available_tickers]

            if len(valid_cols) < 2:
                continue

            # Isolate prices for optimization
            optimization_df = new_df[optimization_start_date:optimization_end_date][price_col][valid_cols]

            success = False
            try:
                # Calculate weights
                lb = round((1 / (len(optimization_df.columns) * 2)), 3)
                weights_dict = optimize_weights(prices=optimization_df, lower_bound=lb)
                weights = pd.DataFrame(weights_dict, index=pd.Series(0))
                success = True
            except Exception as e:
                print(f'Max Sharpe Optimization failed for {start_date}: {e}')

            # Fallback to Equal-Weights
            if not success:
                weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))], 
                                         index=optimization_df.columns.tolist(), 
                                         columns=pd.Series(0)).T

# 5. Performance Calculation
            temp_df = returns_df[start_date:end_date][valid_cols]

            # Stack returns into long format
            returns_long = temp_df.stack().to_frame('return')
            returns_long.index.names = ['date', 'ticker']

            # Stack weights into long format  
            weights_long = weights.stack().to_frame('weight')
            weights_long.index.names = ['row', 'ticker']
            weights_long = weights_long.droplevel('row')

            # Merge on ticker
            returns_long = returns_long.reset_index()
            weights_long = weights_long.reset_index()

            merged = returns_long.merge(weights_long, on='ticker')
            merged['weighted return'] = merged['return'] * merged['weight']

            temp_df = merged.groupby('date')['weighted return'].sum().to_frame('Strategy Return')

            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

        except Exception as e:
            print(f"Error processing {start_date}: {e}")

    return portfolio_df.drop_duplicates()   

# def portfolio_visual(portfolio_df, ticker):
#     # 1. Download the benchmark data
#     # We download from the start of your strategy data to the end
#     spy = yf.download(tickers=ticker, start=portfolio_df.index.min(), end=portfolio_df.index.max(), auto_adjust=False)

#     # Standardize column names to lowercase to avoid KeyErrors
#     if isinstance(spy.columns, pd.MultiIndex):
#         spy.columns = spy.columns.set_levels(spy.columns.levels[0].str.lower(), level=0)
#     else:
#         spy.columns = spy.columns.str.lower()

#     # 2. Isolate the price data safely
#     spy_prices = spy['adj close'] if 'adj close' in spy.columns else spy['close']

#     # 3. Calculate log returns
#     spy_ret = np.log(spy_prices).diff().dropna()

#     # 4. Handle Type (Fixes the AttributeError)
#     if isinstance(spy_ret, pd.Series):
#         spy_ret = spy_ret.to_frame('Market Benchmark')
#     else:
#         spy_ret.columns = ['Market Benchmark']

#     # 5. Resample to Monthly (Fixes the Empty Chart)
#     # This sums daily log returns into monthly ones to match your strategy
#     spy_ret = spy_ret.resample('ME').sum()

#     # 6. Ensure indexes are both Datetime objects for a clean merge
#     portfolio_df.index = pd.to_datetime(portfolio_df.index).to_period('M').to_timestamp('M')
#     spy_ret.index = pd.to_datetime(spy_ret.index).to_period('M').to_timestamp('M')

#     # 7. Final Merge (Use 'left' join to keep all strategy data)
#     portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True, how='left')

#     return portfolio_df

def portfolio_visual(portfolio_df, ticker):
    
    # Resample daily strategy returns to monthly
    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    portfolio_df = portfolio_df.resample('ME').sum()

    start = portfolio_df.index.min()
    end = portfolio_df.index.max()

    spy = yf.download(tickers=ticker, start=start, end=end, auto_adjust=False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.set_levels(spy.columns.levels[0].str.lower(), level=0)
    else:
        spy.columns = spy.columns.str.lower()

    spy_prices = spy['adj close'] if 'adj close' in spy.columns else spy['close']
    spy_prices = spy_prices.squeeze()

    # Daily log returns resampled to monthly
    spy_ret = np.log(spy_prices / spy_prices.shift(1)).dropna()
    spy_ret = spy_ret.resample('ME').sum()
    spy_ret = spy_ret.to_frame('Market Benchmark')

    # Align both indexes to month-end
    portfolio_df.index = portfolio_df.index + pd.offsets.MonthEnd(0)
    spy_ret.index = spy_ret.index + pd.offsets.MonthEnd(0)

    portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True, how='inner')

    return portfolio_df

def plot_port(portfolio_df):
    plt.style.use('ggplot')

    # Ensure index is datetime
    portfolio_df.index = pd.to_datetime(portfolio_df.index)

    # Drop any rows where ALL values are NaN
    portfolio_df = portfolio_df.dropna(how='all')

    # Sort by date ascending
    portfolio_df = portfolio_df.sort_index()

    # Rename columns cleanly for the legend
    col_rename = {}
    for col in portfolio_df.columns:
        if 'strategy' in col.lower() or 'weighted' in col.lower():
            col_rename[col] = 'Strategy Return'
        elif 'market' in col.lower() or 'benchmark' in col.lower():
            col_rename[col] = 'Market Benchmark'
    portfolio_df = portfolio_df.rename(columns=col_rename)

    # Cumulative return from log returns
    portfolio_cumulative_return = np.exp(portfolio_df.cumsum()) - 1

    fig, ax = plt.subplots(figsize=(16, 6))
    portfolio_cumulative_return.plot(ax=ax, linewidth=1.5)

    ax.set_title('Unsupervised Learning Trading Strategy Returns Over Time', fontsize=14)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylabel('Return')
    ax.set_xlabel('Date')
    ax.legend(fontsize=11)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    return fig
        






   
   













# %%
