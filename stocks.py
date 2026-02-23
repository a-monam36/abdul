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

# returns a dictionary of the stocks seleceted for each month using cluster 0 

  filtered_df = data[data['cluster' == 0]]

  filtered_df = filtered_df.reset_index(level=1)

  filtered_df = filtered_df.index + pd.DateOffset(1)

  filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

  dates = filtered_df.index.get_level_values('date').unique().tolist()

  fixed_dates = {}

  for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()

  return fixed_dates




def portfolio_optimization(data, fixed_dates):

    def optimize_weights(prices, lower_bound=0):
        # Calculate annualised expected returns and the annualised sample covariance matrix
        returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
        covariance = risk_models.sample_cov(prices=prices, frequency=252)
        
        # Define the optimization problem with weight constraints (min: lower_bound, max: 10%)
        ef = EfficientFrontier(expected_returns=returns, cov_matrix=covariance, weight_bounds=(lower_bound, 0.1), solver='SCS')
        
        # Optimize for the Maximum Sharpe Ratio
        weights = ef.max_sharpe()
        return ef.clean_weights()

    # 1. Prepare Tickers and Date Range
    stocks = data.index.get_level_values('ticker').unique().tolist()
    start = data.index.get_level_values('date').unique()[0] - pd.DateOffset(months=12)
    end = data.index.get_level_values('date').unique()[-1]
    
    # 2. Download Data
    new_df = yf.download(tickers=stocks, start=start, end=end)


    returns_df = np.log(new_df['Adj Close']).diff()
    
    portfolio_df = pd.DataFrame()

   
    for start_date in fixed_dates.keys():
        try:
            # Setup current trading month and look-back period
            end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
            cols = fixed_dates[start_date]
            
            optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
            optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')
            
            # Isolate the prices for optimization
            optimization_df = new_df[optimization_start_date:optimization_end_date]['Adj Close'][cols]

            success = False
            try:
                # Calculate weights using the Max Sharpe function
                lb = round((1 / (len(optimization_df.columns) * 2)), 3)
                weights = optimize_weights(prices=optimization_df, lower_bound=lb)
                weights = pd.DataFrame(weights, index=pd.Series(0))
                success = True
            except Exception as e:
                print(f'Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights. Error: {e}')

            # Fallback to Equal-Weights if optimization fails
            if not success:
                weights = pd.DataFrame([1/len(optimization_df.columns) for i in range(len(optimization_df.columns))], 
                                         index=optimization_df.columns.tolist(), 
                                         columns=pd.Series(0)).T

            # 5. Performance Calculation
            temp_df = returns_df[start_date:end_date][cols]
            
            # Align weights and returns via Stacking and Merging
            temp_df = temp_df.stack().to_frame('return').reset_index(level=0) \
                       .merge(weights.stack().to_frame('weight').reset_index(level=0, drop=True), 
                              left_index=True, 
                              right_index=True) \
                       .reset_index().set_index(['Date', 'index']).unstack().stack()

            temp_df.index.names = ['date', 'ticker']
            
            # Calculate daily weighted return
            temp_df['weighted return'] = temp_df['return'] * temp_df['weight']

            # Sum all individual stock returns to get the single daily Strategy Return
            temp_df = temp_df.groupby(level=0)['weighted return'].sum().to_frame('Strategy Return')

            # Append to the master portfolio dataframe
            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)

        except Exception as e:
            print(f"Error processing {start_date}: {e}")

    portfolio_df = portfolio_df.drop_duplicates()
    return portfolio_df
       
       

def portfolio_visual(portfolio_df):
  spy = yf.download(tickers='SPY', start='2010-01-01', end=dt.date.today())
  spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close': 'SPY Buy&Hold'}, axis=1)

  portfolio_df = portfolio_df.merge(spy_ret, left_index= True, right_index= True)

  return portfolio_df

def plot_port(portfolio_df):
  plt.style.use('ggplot')
  

# convert back from log using exp -1 for proper perncentage
  portfolio_cumulative_return = np.exp(portfolio_df.cumsum()) -1
  portfolio_cumulative_return[:'2023-09-29'].plot(figsize=(16,6))
  plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

  #
  plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
  plt.ylabel('Return')
  plt.show()



        






   
   










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
