from yahoo_fin import stock_info as si 
import pandas as pd
import yfinance as yf
import datetime as dt 
from sklearn.linear_model import LinearRegression
import numpy as np

sp_500 = si.tickers_sp500()
date_today = dt.date.today()


def find_potential_pairs(symbol_list):
    prices = yf.download(symbol_list, start=(date_today - dt.timedelta(days=1825)), end=date_today)['Adj Close']
    prices=prices.dropna(axis=1)
#    col_count=prices.shape[1]
#    row_count = prices.shape[0]
#    print(col_count,row_count)

    # Calculate correlation matrix
    corr_matrix = prices.corr()

    pairs=[]
    # Find pairs with high correlation
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.95:  # You can adjust the correlation threshold as needed
                stock1 = yf.download(corr_matrix.columns[i], start=(date_today - dt.timedelta(days=1825)), end=date_today)
                stock2 = yf.download(corr_matrix.columns[j], start=(date_today - dt.timedelta(days=1825)), end=date_today)
                model = LinearRegression()
                model.fit(stock1['Close'].values.reshape(-1, 1), stock2['Close'].values.reshape(-1, 1))
                spread = stock2['Close'] - model.predict(stock1['Close'].values.reshape(-1, 1)).flatten()
                mean_spread = np.mean(spread)
                std_spread = np.std(spread)
                print(mean_spread, std_spread)
                if mean_spread > 0 and std_spread < 2:
#               if mean_spread > 0 and std_spread < 0.05:
                    pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                    
    return pairs

#potential_pairs = find_potential_pairs(sp_500)
#print(potential_pairs)

#OUTPUT
#[('AVB', 'UDR'), ('BAC', 'FITB'), ('CFG', 'HBAN'), ('CFG', 'KEY'), ('CVX', 'MRO'), ('EOG', 'MRO'), ('FANG', 'MRO'), ('GOOG', 'GOOGL'), ('NWS', 'NWSA'), ('PXD', 'WMB'), ('TRGP', 'WMB')]
#REMOVING GOOG, GOOGL
pairs = [('AVB', 'UDR'), ('BAC', 'FITB'), ('CFG', 'HBAN'), ('CFG', 'KEY'), ('CVX', 'MRO'), ('EOG', 'MRO'), ('FANG', 'MRO'), ('NWS', 'NWSA'), ('PXD', 'WMB'), ('TRGP', 'WMB')]
max_marketcap = 0
max_market_pair = ()
for i in pairs:
    try:
        marketcap = yf.Ticker(i[0]).basic_info['marketCap'] + yf.Ticker(i[1]).basic_info['marketCap']
        if marketcap > max_marketcap:
            max_marketcap = marketcap
            max_market_pair = i
    except:
        pass

print(max_market_pair)

#OUTPUT: ('CVX', 'MRO')
#Analyzing on ('CVX', 'MRO')