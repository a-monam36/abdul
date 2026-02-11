import pandas as pd
import yfinance as yf
import pandas_ta_classic as ta
# import pandas_ta as ta
import pypfopt
import sklearn
import matplotlib.pyplot as plt

print("--- Environment Health Check ---")
print(f"Pandas Version: {pd.__version__}")
print(f"YFinance Version: {yf.__version__}")
print(f"Scikit-Learn Version: {sklearn.__version__}")
print("--------------------------------")
print("✅ All libraries loaded successfully!")