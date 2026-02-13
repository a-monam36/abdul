def compute_macd():
    macd = pandas_ta.macd(close= close, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)



df['dollar volume'] = (df['adj close']*df['volume'] ) /1e6



def compute_bbands(x):
    bands = pandas_ta.bbands(close= np.log1p(x), length=20)

    return bands.iloc[:, 0:3]

bbands_df = df.groupby(level=1,  group_keys= False)['adj close'].apply(compute_bbands)

bbands_df.columns = ['bb_low', 'bb_mid', 'bb_high']

df = df.join(bbands_df)

df = df.dropna()