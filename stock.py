# CHECK PANDAS TA LIBRARY, LOOKS POWERFUL AND CAN REPLACE TALIB FUNCTIONS

import streamlit as st 
import hydralit_components as hc
import datetime
import pandas as pd 
import yfinance as yf
import numpy as np
import pandas_datareader as pdr
import mplfinance as fplt
import backtrader as bt 
import matplotlib.pyplot as plt
import talib
import matplotlib
import requests
import tweepy
import plotly.graph_objs as go
from yahoo_fin import news as nws 
from pandas.tseries.offsets import BDay
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs
from streamlit_option_menu import option_menu
from string import Template
from datetime import date, timedelta
from yahoo_fin import stock_info as si 
from pandas_datareader import data as pdr
import os
print(si.get_earnings('TSLA'))
'''
import pypfopt #AttributeError: type object 'spmatrix' has no attribute '__div__'
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices #discrete AttributeError: type object 'spmatrix' has no attribute '__div__'
from pypfopt import EfficientFrontier #discrete AttributeError: type object 'spmatrix' has no attribute '__div__'
from pypfopt import risk_models #discrete AttributeError: type object 'spmatrix' has no attribute '__div__'
from pypfopt import expected_returns #discrete AttributeError: type object 'spmatrix' has no attribute '__div__'
from pypfopt import plotting #discrete AttributeError: type object 'spmatrix' has no attribute '__div__'
from pypfopt import objective_functions #discrete AttributeError: type object 'spmatrix' has no attribute '__div__' 
import plotly.express as px
import copy
from datetime import datetime
from io import BytesIO
from yahooquery import Screener
from nltk.sentiment.vader import SentimentIntensityAnalyzer

print (yf.Ticker('TSLA').get_actions)
print (yf.Ticker('TSLA').get_analyst_price_target)
print (yf.Ticker('TSLA').get_balance_sheet)
print (yf.Ticker('TSLA').get_balancesheet)
print (yf.Ticker('TSLA')._get_ticker_tz) #['currency', 'dayHigh', 'dayLow', 'exchange', 'fiftyDayAverage', 'lastPrice', 'lastVolume', 'marketCap', 'open', 'previousClose', 'quoteType', 'regularMarketPreviousClose', 'shares', 'tenDayAverageVolume', 'threeMonthAverageVolume', 'timezone', 'twoHundredDayAverage', 'yearChange', 'yearHigh', 'yearLow']
print (yf.Ticker('TSLA').get_calendar)
print (yf.Ticker('TSLA').get_capital_gains)
print (yf.Ticker('TSLA').get_cash_flow)
print (yf.Ticker('TSLA').get_cashflow)
print (yf.Ticker('TSLA').get_dividends)
print (yf.Ticker('TSLA').get_earnings)
print (yf.Ticker('TSLA').get_earnings_dates)
print (yf.Ticker('TSLA').get_earnings_forecast)
print (yf.Ticker('TSLA').get_earnings_trend)
print (yf.Ticker('TSLA').get_fast_info)
print (yf.Ticker('TSLA').get_financials)
print (yf.Ticker('TSLA').get_history_metadata)
print (yf.Ticker('TSLA').get_income_stmt)
print (yf.Ticker('TSLA').get_incomestmt)
print (yf.Ticker('TSLA').get_info)
print (yf.Ticker('TSLA').get_institutional_holders)
print (yf.Ticker('TSLA').get_isin)
print (yf.Ticker('TSLA').get_major_holders)
print (yf.Ticker('TSLA').get_mutualfund_holders)
print (yf.Ticker('TSLA').get_news)
print (yf.Ticker('TSLA').get_recommendations)
print (yf.Ticker('TSLA').get_recommendations_summary)
print (yf.Ticker('TSLA').get_rev_forecast)
print (yf.Ticker('TSLA').get_shares)
print (yf.Ticker('TSLA').get_splits)
print (yf.Ticker('TSLA').get_sustainability)
print (yf.Ticker('TSLA').get_trend_details)

print (yf.Ticker('TSLA').actions)
#print (yf.Ticker('TSLA').analyst_price_target)
print (yf.Ticker('TSLA').balance_sheet)
print (yf.Ticker('TSLA').balancesheet)
print (yf.Ticker('TSLA').basic_info) #['currency', 'dayHigh', 'dayLow', 'exchange', 'fiftyDayAverage', 'lastPrice', 'lastVolume', 'marketCap', 'open', 'previousClose', 'quoteType', 'regularMarketPreviousClose', 'shares', 'tenDayAverageVolume', 'threeMonthAverageVolume', 'timezone', 'twoHundredDayAverage', 'yearChange', 'yearHigh', 'yearLow']
#print (yf.Ticker('TSLA').calendar)
print (yf.Ticker('TSLA').capital_gains)
print (yf.Ticker('TSLA').cash_flow)
print (yf.Ticker('TSLA').cashflow)
print (yf.Ticker('TSLA').dividends)
#print (yf.Ticker('TSLA').earnings)
print (yf.Ticker('TSLA').earnings_dates)
#print (yf.Ticker('TSLA').earnings_forecasts)
#print (yf.Ticker('TSLA').earnings_trend)
print (yf.Ticker('TSLA').fast_info)
print (yf.Ticker('TSLA').financials)
print (yf.Ticker('TSLA').history_metadata)
print (yf.Ticker('TSLA').income_stmt)
print (yf.Ticker('TSLA').incomestmt)
print (yf.Ticker('TSLA').info)
print (yf.Ticker('TSLA').institutional_holders)
print (yf.Ticker('TSLA').isin)
print (yf.Ticker('TSLA').major_holders)
print (yf.Ticker('TSLA').mutualfund_holders)
print (yf.Ticker('TSLA').news)
print (yf.Ticker('TSLA').options)
print (yf.Ticker('TSLA').quarterly_balance_sheet)
print (yf.Ticker('TSLA').quarterly_balancesheet)
print (yf.Ticker('TSLA').quarterly_cash_flow)
print (yf.Ticker('TSLA').quarterly_cashflow)
#print (yf.Ticker('TSLA').quarterly_earnings)
print (yf.Ticker('TSLA').quarterly_financials)
print (yf.Ticker('TSLA').quarterly_income_stmt)
print (yf.Ticker('TSLA').quarterly_incomestmt)
#print (yf.Ticker('TSLA').recommendations)
#print (yf.Ticker('TSLA').recommendations_summary)
#print (yf.Ticker('TSLA').revenue_forecasts)
#print (yf.Ticker('TSLA').shares)
print (yf.Ticker('TSLA').splits)
#print (yf.Ticker('TSLA').sustainability)
#print (yf.Ticker('TSLA').trend_details)

stock_data= yf.download('TSLA')
stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
market_data = yf.download('^GSPC')  # S&P 500 index as the market benchmark
market_data['Daily_Return'] = market_data['Adj Close'].pct_change()
# Calculate historical volatility (standard deviation of daily returns)
stock_volatility = stock_data['Daily_Return'].std()
market_volatility = market_data['Daily_Return'].std()

cov_stock = stock_data['Daily_Return'].cov(market_data['Daily_Return'])
var_market = market_data['Daily_Return'].var()

beta_stock = cov_stock / var_market
risk_free_rate = 0.02

actual_return = stock_data['Daily_Return'].mean()
expected_return =  risk_free_rate + beta_stock * (market_data['Daily_Return'].mean() - risk_free_rate)
alpha= actual_return - expected_return
print(beta_stock)
print(alpha)

print(nws.get_yf_rss('TSLA'))
print(si.build_url('TSLA'))
print(si.get_analysts_info('TSLA'))
print(si.get_balance_sheet('TSLA'))
#print(si.get_cash_flow('TSLA'))
#print(si.get_company_info('TSLA'))
#print(si.get_company_officers('TSLA'))
print(si.get_currencies())
print(si.get_data('TSLA'))
print(si.get_day_gainers(5))
print(si.get_day_losers(5))
print(si.get_day_most_active())
print(si.get_dividends('TSLA'))
print(si.get_earnings('TSLA'))
#print(si.get_earnings_for_date(date='2023-12-01'))
#print(si.get_earnings_history('TSLA'))
#print(si.get_earnings_in_date_range(start_date='2023-11-01', end_date='2023-12-01'))
#print(si.get_financials('TSLA'))
print(si.get_futures())
print(si.get_holders('TSLA'))
#print(si.get_income_statement('TSLA'))
print(si.get_live_price('TSLA'))
#print(si.get_market_status())
#print(si.get_next_earnings_date('TSLA'))
#print(si.get_postmarket_price('TSLA'))
#print(si.get_premarket_price('TSLA'))
#print(si.get_quote_data('AAPL'))
#print(si.get_quote_table('TSLA'))
print(si.get_splits('TSLA'))
#print(si.get_stats('TSLA'))
#print(si.get_stats_valuation('TSLA'))
#print(si.get_top_crypto())
print(si.get_undervalued_large_caps())
print(si.tickers_dow())
#print(si.tickers_ftse100())
#print(si.tickers_ftse250())
#print(si.tickers_ibovespa())
print(si.tickers_nasdaq())
print(si.tickers_nifty50())
print(si.tickers_niftybank())
print(si.tickers_other())
print(si.tickers_sp500())


import pandas_datareader.data as web
import pandas_datareader as wb
import statsmodels.formula.api as smf
print(pdr.BankOfCanadaReader('FXUSDCAD').read())

# WORLDBANK
#matches = wb.search('gdp.*capita.*const')
#print(matches)
#dat = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CA', 'MX'], start=2005, end=2008)
#print(dat)
#print(wb.search('cell.*%').iloc[:,:2])
#ind = ['NY.GDP.PCAP.KD', 'IT.MOB.COV.ZS']
#dat = wb.download(indicator=ind, country='all', start=2011, end=2011).dropna()
#dat.columns = ['gdp', 'cellphone']
#print(dat.tail())
#mod = smf.ols('cellphone ~ np.log(gdp)', dat).fit()
#print(mod.summary())

#ALPHAVANTAGE
#print(web.get_sector_performance_av().head())
#print(web.DataReader("USD/JPY", "av-forex",  api_key=os.getenv('ALPHAVANTAGE_API_KEY')))

#FRED
rate=pdr.DataReader('DGS10', 'fred') #10-year U.S. Treasury yield 
print(rate)
gdp = web.DataReader('GDP', 'fred')
print(gdp)
inflation = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred')
print(inflation)

#FAMAFRENCH
ds = web.DataReader('5_Industry_Portfolios', 'famafrench')
print(ds)

#ECONDB
#print(web.DataReader('ticker=RGDPUS', 'econdb').head())
#print(web.DataReader('dataset=NAMQ_10_GDP&v=Geopolitical entity (reporting)&h=TIME&from=2018-05-01&to=2021-01-01&GEO=[AL,AT,BE,BA,BG,HR,CY,CZ,DK,EE,EA19,FI,FR,DE,EL,HU,IS,IE,IT,XK,LV,LT,LU,MT,ME,NL,MK,NO,PL,PT,RO,RS,SK,SI,ES,SE,CH,TR,UK]&NA_ITEM=[B1GQ]&S_ADJ=[SCA]&UNIT=[CLV10_MNAC]', 'econdb').columns)

#OECD 
df = web.DataReader('TUD', 'oecd')
print(df[['Japan', 'United States']])

#EUROSTAT
#df = web.DataReader('tran_sf_railac', 'eurostat')
#print(df)

#Thrift Savings Plan (TSP) Fund Data
import pandas_datareader.tsp as tsp
#print(tsp.TSPReader(start='2015-10-1', end='2015-12-31').read())

#Nasdaq Trader Symbol Definitions
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
symbols = get_nasdaq_symbols()
print(symbols.loc['IBM'])

# yahoo
#actions = web.DataReader('GOOG', 'yahoo-actions')
#print(actions.head())
#dividends = web.DataReader('IBM', 'yahoo-dividends')
#print(dividends.head())

indicators_dict = {
    "AD": "Chaikin A/D Line",
    "ADOSC": "Chaikin A/D Oscillator",
    "ADX": "Average Directional Movement Index",
    "ADXR": "Average Directional Movement Index Rating",
    "APO": "Absolute Price Oscillator",
    "AROON": "Aroon",
    "AROONOSC": "Aroon Oscillator",
    "ATR": "Average True Range",
    "AVGPRICE": "Average Price",
    "BBANDS": "Bollinger Bands",
    "BETA": "Beta",
    "BOP": "Balance Of Power",
    "CCI": "Commodity Channel Index",
    "CDL2CROWS": "Two Crows",
    "CDL3BLACKCROWS": "Three Black Crows",
    "CDL3INSIDE": "Three Inside Up/Down",
    "CDL3LINESTRIKE": "Three Outside Up/Down",
    "CDL3STARSINSOUTH": "Three Stars In The South",
    "CDL3WHITESOLDIERS": "Three Advancing White Soldiers",
    "CDLABANDONEDBABY": "Abandoned Baby",
    "CDLADVANCEBLOCK": "Advance Block",
    "CDLBELTHOLD": "Belt-hold",
    "CDLBREAKAWAY": "Breakaway",
    "CDLCLOSINGMARUBOZU": "Closing Marubozu",
    "CDLCONCEALBABYSWALL": "Concealing Baby Swallow",
    "CDLCOUNTERATTACK": "Counterattack",
    "CDLDARKCLOUDCOVER": "Dark Cloud Cover",
    "CDLDOJI": "Doji",
    "CDLDOJISTAR": "Doji Star",
    "CDLDRAGONFLYDOJI": "Dragonfly Doji",
    "CDLENGULFING": "Engulfing Pattern",
    "CDLEVENINGDOJISTAR": "Evening Doji Star",
    "CDLEVENINGSTAR": "Evening Star",
    "CDLGAPSIDESIDEWHITE": "Up/Down-gap side-by-side white lines",
    "CDLGRAVESTONEDOJI": "Gravestone Doji",
    "CDLHAMMER": "Hammer",
    "CDLHANGINGMAN": "Hanging Man",
    "CDLHARAMI": "Harami Pattern",
    "CDLHARAMICROSS": "Harami Cross Pattern",
    "CDLHIGHWAVE": "High-Wave Candle",
    "CDLHIKKAKE": "Hikkake Pattern",
    "CDLHIKKAKEMOD": "Modified Hikkake Pattern",
    "CDLHOMINGPIGEON": "Homing Pigeon",
    "CDLIDENTICAL3CROWS": "Identical Three Crows",
    "CDLINNECK": "In-Neck Pattern",
    "CDLINVERTEDHAMMER": "Inverted Hammer",
    "CDLKICKING": "Kicking",
    "CDLKICKINGBYLENGTH": "Kicking - bull/bear determined by the longer marubozu",
    "CDLLADDERBOTTOM": "Ladder Bottom",
    "CDLLONGLEGGEDDOJI": "Long Legged Doji",
    "CDLLONGLINE": "Long Line Candle",
    "CDLMARUBOZU": "Marubozu",
    "CDLMATCHINGLOW": "Matching Low",
    "CDLMATHOLD": "Mat Hold",
    "CDLMORNINGDOJISTAR": "Morning Doji Star",
    "CDLMORNINGSTAR": "Morning Star",
    "CDLONNECK": "On-Neck Pattern",
    "CDLPIERCING": "Piercing Pattern",
    "CDLRICKSHAWMAN": "Rickshaw Man",
    "CDLRISEFALL3METHODS": "Rising/Falling Three Methods",
    "CDLSEPARATINGLINES": "Separating Lines",
    "CDLSHOOTINGSTAR": "Shooting Star",
    "CDLSHORTLINE": "Short Line Candle",
    "CDLSPINNINGTOP": "Spinning Top",
    "CDLSTALLEDPATTERN": "Stalled Pattern",
    "CDLSTICKSANDWICH": "Stick Sandwich",
    "CDLTAKURI": "Takuri (Dragonfly Doji with very long lower shadow)",
    "CDLTASUKIGAP": "Tasuki Gap",
    "CDLTHRUSTING": "Thrusting Pattern",
    "CDLTRISTAR": "Tristar Pattern",
    "CDLUNIQUE3RIVER": "Unique 3 River",
    "CDLUPSIDEGAP2CROWS": "Upside Gap Two Crows",
    "CDLXSIDEGAP3METHODS": "Upside/Downside Gap Three Methods",
    "CMO": "Chande Momentum Oscillator",
    "CORREL": "Pearson's Correlation Coefficient (r)",
    "DEMA": "Double Exponential Moving Average",
    "DX": "Directional Movement Index",
    "EMA": "Exponential Moving Average",
    "HT_DCPERIOD": "Hilbert Transform - Dominant Cycle Period",
    "HT_DCPHASE": "Hilbert Transform - Dominant Cycle Phase",
    "HT_PHASOR": "Hilbert Transform - Phasor Components",
    "HT_SINE": "Hilbert Transform - SineWave",
    "HT_TRENDLINE": "Hilbert Transform - Instantaneous Trendline",
    "HT_TRENDMODE": "Hilbert Transform - Trend vs Cycle Mode",
    "KAMA": "Kaufman Adaptive Moving Average",
    "LINEARREG": "Linear Regression",
    "LINEARREG_ANGLE": "Linear Regression Angle",
    "LINEARREG_INTERCEPT": "Linear Regression Intercept",
    "LINEARREG_SLOPE": "Linear Regression Slope",
    "MA": "All Moving Average",
    "MACD": "Moving Average Convergence/Divergence",
    "MACDEXT": "MACD with controllable MA type",
    "MACDFIX": "Moving Average Convergence/Divergence Fix 12/26",
    "MAMA": "MESA Adaptive Moving Average",
    "MAX": "Highest value over a specified period",
    "MAXINDEX": "Index of highest value over a specified period",
    "MEDPRICE": "Median Price",
    "MFI": "Money Flow Index",
    "MIDPOINT": "MidPoint over period",
    "MIDPRICE": "Midpoint Price over period",
    "MIN": "Lowest value over a specified period",
    "MININDEX": "Index of lowest value over a specified period",
    "MINMAX": "Lowest and highest values over a specified period",
    "MINMAXINDEX": "Indexes of lowest and highest values over a specified period",
        "MINUS_DI": "Minus Directional Indicator",
    "MINUS_DM": "Minus Directional Movement",
    "MOM": "Momentum",
    "NATR": "Normalized Average True Range",
    "OBV": "On Balance Volume",
    "PLUS_DI": "Plus Directional Indicator",
    "PLUS_DM": "Plus Directional Movement",
    "PPO": "Percentage Price Oscillator",
    "ROC": "Rate of change : ((price/prevPrice)-1)*100",
    "ROCP": "Rate of change Percentage: (price-prevPrice)/prevPrice",
    "ROCR": "Rate of change ratio: (price/prevPrice)",
    "ROCR100": "Rate of change ratio 100 scale: (price/prevPrice)*100",
    "RSI": "Relative Strength Index",
    "SAR": "Parabolic SAR",
    "SAREXT": "Parabolic SAR - Extended",
    "SMA": "Simple Moving Average",
    "STDDEV": "Standard Deviation",
    "STOCH": "Stochastic",
    "STOCHF": "Stochastic Fast",
    "STOCHRSI": "Stochastic Relative Strength Index",
    "SUM": "Summation",
    "T3": "Triple Exponential Moving Average (T3)",
    "TEMA": "Triple Exponential Moving Average",
    "TRANGE": "True Range",
    "TRIMA": "Triangular Moving Average",
    "TRIX": "1-day Rate-Of-Change (ROC) of a Triple Smooth EMA",
    "TSF": "Time Series Forecast",
    "TYPPRICE": "Typical Price",
    "ULTOSC": "Ultimate Oscillator",
    "VAR": "Variance",
    "WCLPRICE": "Weighted Close Price",
    "WILLR": "Williams' %R",
    "WMA": "Weighted Moving Average"
}
  
for i in indicators_dict:
    print("talib."+i+"(df_needed, time_needed)")

talib.AD(df_needed, time_needed)
talib.ADOSC(df_needed, time_needed)
talib.ADX(df_needed, time_needed)
talib.ADXR(df_needed, time_needed)
talib.APO(df_needed, time_needed)
talib.AROON(df_needed, time_needed)
talib.AROONOSC(df_needed, time_needed)
talib.ATR(df_needed, time_needed)
talib.AVGPRICE(df_needed, time_needed)
talib.BBANDS(df_needed, time_needed)
talib.BETA(df_needed, time_needed)
talib.BOP(df_needed, time_needed)
talib.CCI(df_needed, time_needed)
talib.CDL2CROWS(df_needed, time_needed)
talib.CDL3BLACKCROWS(df_needed, time_needed)
talib.CDL3INSIDE(df_needed, time_needed)
talib.CDL3LINESTRIKE(df_needed, time_needed)
talib.CDL3STARSINSOUTH(df_needed, time_needed)
talib.CDL3WHITESOLDIERS(df_needed, time_needed)
talib.CDLABANDONEDBABY(df_needed, time_needed)
talib.CDLADVANCEBLOCK(df_needed, time_needed)
talib.CDLBELTHOLD(df_needed, time_needed)
talib.CDLBREAKAWAY(df_needed, time_needed)
talib.CDLCLOSINGMARUBOZU(df_needed, time_needed)
talib.CDLCONCEALBABYSWALL(df_needed, time_needed)
talib.CDLCOUNTERATTACK(df_needed, time_needed)
talib.CDLDARKCLOUDCOVER(df_needed, time_needed)
talib.CDLDOJI(df_needed, time_needed)
talib.CDLDOJISTAR(df_needed, time_needed)
talib.CDLDRAGONFLYDOJI(df_needed, time_needed)
talib.CDLENGULFING(df_needed, time_needed)
talib.CDLEVENINGDOJISTAR(df_needed, time_needed)
talib.CDLEVENINGSTAR(df_needed, time_needed)
talib.CDLGAPSIDESIDEWHITE(df_needed, time_needed)
talib.CDLGRAVESTONEDOJI(df_needed, time_needed)
talib.CDLHAMMER(df_needed, time_needed)
talib.CDLHANGINGMAN(df_needed, time_needed)
talib.CDLHARAMI(df_needed, time_needed)
talib.CDLHARAMICROSS(df_needed, time_needed)
talib.CDLHIGHWAVE(df_needed, time_needed)
talib.CDLHIKKAKE(df_needed, time_needed)
talib.CDLHIKKAKEMOD(df_needed, time_needed)
talib.CDLHOMINGPIGEON(df_needed, time_needed)
talib.CDLIDENTICAL3CROWS(df_needed, time_needed)
talib.CDLINNECK(df_needed, time_needed)
talib.CDLINVERTEDHAMMER(df_needed, time_needed)
talib.CDLKICKING(df_needed, time_needed)
talib.CDLKICKINGBYLENGTH(df_needed, time_needed)
talib.CDLLADDERBOTTOM(df_needed, time_needed)
talib.CDLLONGLEGGEDDOJI(df_needed, time_needed)
talib.CDLLONGLINE(df_needed, time_needed)
talib.CDLMARUBOZU(df_needed, time_needed)
talib.CDLMATCHINGLOW(df_needed, time_needed)
talib.CDLMATHOLD(df_needed, time_needed)
talib.CDLMORNINGDOJISTAR(df_needed, time_needed)
talib.CDLMORNINGSTAR(df_needed, time_needed)
talib.CDLONNECK(df_needed, time_needed)
talib.CDLPIERCING(df_needed, time_needed)
talib.CDLRICKSHAWMAN(df_needed, time_needed)
talib.CDLRISEFALL3METHODS(df_needed, time_needed)
talib.CDLSEPARATINGLINES(df_needed, time_needed)
talib.CDLSHOOTINGSTAR(df_needed, time_needed)
talib.CDLSHORTLINE(df_needed, time_needed)
talib.CDLSPINNINGTOP(df_needed, time_needed)
talib.CDLSTALLEDPATTERN(df_needed, time_needed)
talib.CDLSTICKSANDWICH(df_needed, time_needed)
talib.CDLTAKURI(df_needed, time_needed)
talib.CDLTASUKIGAP(df_needed, time_needed)
talib.CDLTHRUSTING(df_needed, time_needed)
talib.CDLTRISTAR(df_needed, time_needed)
talib.CDLUNIQUE3RIVER(df_needed, time_needed)
talib.CDLUPSIDEGAP2CROWS(df_needed, time_needed)
talib.CDLXSIDEGAP3METHODS(df_needed, time_needed)
talib.CMO(df_needed, time_needed)
talib.CORREL(df_needed, time_needed)
talib.DEMA(df_needed, time_needed)
talib.DX(df_needed, time_needed)
talib.EMA(df_needed, time_needed)
talib.HT_DCPERIOD(df_needed, time_needed)
talib.HT_DCPHASE(df_needed, time_needed)
talib.HT_PHASOR(df_needed, time_needed)
talib.HT_SINE(df_needed, time_needed)
talib.HT_TRENDLINE(df_needed, time_needed)
talib.HT_TRENDMODE(df_needed, time_needed)
talib.KAMA(df_needed, time_needed)
talib.LINEARREG(df_needed, time_needed)
talib.LINEARREG_ANGLE(df_needed, time_needed)
talib.LINEARREG_INTERCEPT(df_needed, time_needed)
talib.LINEARREG_SLOPE(df_needed, time_needed)
talib.MA(df_needed, time_needed)
talib.MACD(df_needed, time_needed)
talib.MACDEXT(df_needed, time_needed)
talib.MACDFIX(df_needed, time_needed)
talib.MAMA(df_needed, time_needed)
talib.MAX(df_needed, time_needed)
talib.MAXINDEX(df_needed, time_needed)
talib.MEDPRICE(df_needed, time_needed)
talib.MFI(df_needed, time_needed)
talib.MIDPOINT(df_needed, time_needed)
talib.MIDPRICE(df_needed, time_needed)
talib.MIN(df_needed, time_needed)
talib.MININDEX(df_needed, time_needed)
talib.MINMAX(df_needed, time_needed)
talib.MINMAXINDEX(df_needed, time_needed)
talib.MINUS_DI(df_needed, time_needed)
talib.MINUS_DM(df_needed, time_needed)
talib.MOM(df_needed, time_needed)
talib.NATR(df_needed, time_needed)
talib.OBV(df_needed, time_needed)
talib.PLUS_DI(df_needed, time_needed)
talib.PLUS_DM(df_needed, time_needed)
talib.PPO(df_needed, time_needed)
talib.ROC(df_needed, time_needed)
talib.ROCP(df_needed, time_needed)
talib.ROCR(df_needed, time_needed)
talib.ROCR100(df_needed, time_needed)
talib.RSI(df_needed, time_needed)
talib.SAR(df_needed, time_needed)
talib.SAREXT(df_needed, time_needed)
talib.SMA(df_needed, time_needed)
talib.STDDEV(df_needed, time_needed)
talib.STOCH(df_needed, time_needed)
talib.STOCHF(df_needed, time_needed)
talib.STOCHRSI(df_needed, time_needed)
talib.SUM(df_needed, time_needed)
talib.T3(df_needed, time_needed)
talib.TEMA(df_needed, time_needed)
talib.TRANGE(df_needed, time_needed)
talib.TRIMA(df_needed, time_needed)
talib.TRIX(df_needed, time_needed)
talib.TSF(df_needed, time_needed)
talib.TYPPRICE(df_needed, time_needed)
talib.ULTOSC(df_needed, time_needed)
talib.VAR(df_needed, time_needed)
talib.WCLPRICE(df_needed, time_needed)
talib.WILLR(df_needed, time_needed)
talib.WMA(df_needed, time_needed)

import investpy as ip 
print(ip.get_commodities())
print(ip.get_available_currencies())
print(ip.get_bond_countries())
#print(ip.get_bond_historical_data(bond from end))
#print(ip.get_bond_information(bond))
#print(ip.get_bond_recent_data(bond))
print(ip.get_bonds())
#print(ip.get_bonds_overview(country))
#print(ip.get_certificate_countries(certificate country from to))
#print(ip.get_certificate_historical_data(certificate country from to))
#print(ip.get_certificate_information(certificate country))
#print(ip.get_certificate_recent_data(certificate country))
print(ip.get_certificates())
#print(ip.get_certificates_overview(country))
print(ip.get_commodities())
#print(ip.get_commodities_overview(group))
#print(ip.get_commodity_groups(commodity from to))
#print(ip.get_commodity_historical_data(commodity from to))
#print(ip.get_commodity_information(commodity))
#print(ip.get_commodity_recent_data(commodity))
#print(ip.get_crypto_historical_data(crypto from to))
#print(ip.get_crypto_information(crypto))
#print(ip.get_crypto_recent_data(crypto))
print(ip.get_cryptos())
#print(ip.get_cryptos_overview())
#print(ip.get_currency_cross_historical_data('currency_cross', 'from_date', 'to_date'))
#print(ip.get_currency_cross_information(currency_cross))
#print(ip.get_currency_cross_recent_data(currency_cross))
#print(ip.get_currency_crosses_overview(currency))
#print(ip.get_etf_historical_data(etf country from to))
#print(ip.get_etf_information(etf country))
#print(ip.get_etf_countries(etf country))
print(ip.get_etfs())
#print(ip.get_etf_recent_data(etf country))
#print(ip.get_etfs_overview(country))
#print(ip.get_fund_countries(fund country from to))
#print(ip.get_fund_historical_data(fund country from to))
#print(ip.get_fund_information(fund country))
#print(ip.get_fund_recent_data(fund country))
print(ip.get_funds())
#print(ip.get_funds_overview(country))
print(ip.get_index_countries())
#print(ip.get_index_information(index country))
#print(ip.get_index_historical_data(index country from to))
#print(ip.get_index_recent_data(index country))
print(ip.get_indices())
#print(ip.get_indices_overview(country))
#print(ip.get_index_information(index country))
#print(ip.get_stock_company_profile(stock))
#print(ip.get_stock_dividends(stock country))
print(ip.get_stock_countries())
#print(ip.get_stock_historical_data(stock country from to))
#print(ip.get_stock_financial_summary(stock country))
#print(ip.get_stock_recent_data(stock country))
#print(ip.get_stocks_overview(country))
print(ip.get_stocks())
#print(ip.moving_averages(name='bbva', country='spain', product_type='stock', interval='daily'))
#print(ip.search_bonds())
#print(ip.search_certificates())
#print(ip.search_commodities())
#print(ip.pivot_points(name='bbva', country='spain', product_type='stock', interval='daily'))
#print(ip.search_cryptos('symbol', 'BTC'))
#print(ip.search_currency_crosses())
#print(ip.search_etfs())
#print(ip.search_quotes('BTC'))
#print(ip.technical_indicators)
#print(ip.news.economic_calendar())'''

#print(yf.Ticker('TSLA')._get_ticker_tz())

import investpy as ip 
'''CRYPTO'''
#print(si.get_top_crypto())
print(ip.get_cryptos().iloc[:10])
#print(ip.get_cryptos_overview())
#print(si.get_top_crypto())


print(si.get_undervalued_large_caps().sort_values(by='Market Cap', ascending=False))
print(si.get_undervalued_large_caps().sort_values(by='PE Ratio (TTM)', ascending=False))
print(si.tickers_dow()) #USA TOP 30 STOCKS
#print(si.tickers_ftse100()) #LONDON STOCK EXCHANGE TOP 100 STOCKS
#print(si.tickers_ftse250()) #LONDON STOCK EXCHANGE 101st to the 350th largest companies USUALLY MID CAP
#print(si.tickers_ibovespa()) #BRAZILIAN STOCK EXCHANGE about 86 stocks traded
print(si.tickers_nasdaq()) # US stocks
print(si.tickers_nifty50()) # INDIA STOCKS TOP 50 
print(si.tickers_niftybank()) # INDIA STOCKS ON BANKS ONLY
print(si.tickers_other())
print(si.tickers_sp500()) # US STOCKS

import pandas_datareader.data as web
from pandas_datareader import wb
import statsmodels.formula.api as smf
'''
U.S. Research Returns Data (Downloadable Files)

Changes in CRSP Data

Fama/French 3 Factors  TXT  CSV  Details  Historical Archives
Fama/French 3 Factors [Weekly]  TXT  CSV  Details
Fama/French 3 Factors [Daily]  TXT  CSV  Details

Fama/French 5 Factors (2x3)  TXT  CSV  Details  Historical Archives
Fama/French 5 Factors (2x3) [Daily]  TXT  CSV  Details

Univariate sorts on Size, B/M, OP, and Inv

Portfolios Formed on Size  TXT  CSV  Details
Portfolios Formed on Size [ex.Dividends]  TXT  CSV  Details
Portfolios Formed on Size [Daily]  TXT  CSV  Details

Portfolios Formed on Book-to-Market  TXT  CSV  Details
Portfolios Formed on Book-to-Market [ex. Dividends]  TXT  CSV  Details
Portfolios Formed on Book-to-Market [Daily]  TXT  CSV  Details

Portfolios Formed on Operating Profitability  TXT  CSV  Details
Portfolios Formed on Operating Profitability [ex. Dividends]  TXT  CSV  Details
Portfolios Formed on Operating Profitability [Daily]  TXT  CSV  Details

Portfolios Formed on Investment  TXT  CSV  Details
Portfolios Formed on Investment [ex. Dividends]  TXT  CSV  Details
Portfolios Formed on Investment [Daily]  TXT  CSV  Details
Bivariate sorts on Size, B/M, OP and Inv

6 Portfolios Formed on Size and Book-to-Market (2 x 3)  TXT  CSV  Details  Historical Archives
6 Portfolios Formed on Size and Book-to-Market (2 x 3) [ex. Dividends]  TXT  CSV  Details
6 Portfolios Formed on Size and Book-to-Market (2 x 3) [Weekly]  TXT  CSV  Details
6 Portfolios Formed on Size and Book-to-Market (2 x 3) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Size and Book-to-Market (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Size and Book-to-Market (5 x 5) [ex. Dividends]  TXT  CSV  Details
25 Portfolios Formed on Size and Book-to-Market (5 x 5) [Daily]  TXT  CSV  Details

100 Portfolios Formed on Size and Book-to-Market (10 x 10)  TXT  CSV  Details
100 Portfolios Formed on Size and Book-to-Market (10 x 10) [ex. Dividends]  TXT  CSV  Details
100 Portfolios Formed on Size and Book-to-Market (10 x 10) [Daily]  TXT  CSV  Details

6 Portfolios Formed on Size and Operating Profitability (2 x 3)  TXT  CSV  Details  Historical Archives
6 Portfolios Formed on Size and Operating Profitability (2 x 3) [ex. Dividends]  TXT  CSV  Details
6 Portfolios Formed on Size and Operating Profitability (2 x 3) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Size and Operating Profitability (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Size and Operating Profitability (5 x 5) [ex. Dividends]  TXT  CSV  Details
25 Portfolios Formed on Size and Operating Profitability (5 x 5) [Daily]  TXT  CSV  Details

100 Portfolios Formed on Size and Operating Profitability (10 x 10)  TXT  CSV  Details
100 Portfolios Formed on Size and Operating Profitability (10 x 10) [ex. Dividends]  TXT  CSV  Details
100 Portfolios Formed on Size and Operating Profitability (10 x 10) [Daily]  TXT  CSV  Details

6 Portfolios Formed on Size and Investment (2 x 3)  TXT  CSV  Details  Historical Archives
6 Portfolios Formed on Size and Investment (2 x 3) [ex. Dividends]  TXT  CSV  Details
6 Portfolios Formed on Size and Investment (2 x 3) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Size and Investment (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Size and Investment (5 x 5) [ex. Dividends]  TXT  CSV  Details
25 Portfolios Formed on Size and Investment (5 x 5) [Daily]  TXT  CSV  Details

100 Portfolios Formed on Size and Investment (10 x 10)  TXT  CSV  Details
100 Portfolios Formed on Size and Investment (10 x 10) [ex. Dividends]  TXT  CSV  Details
100 Portfolios Formed on Size and Investment (10 x 10) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Book-to-Market and Operating Profitability (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Book-to-Market and Operating Profitability (5 x 5) [ex. Dividends]  TXT  CSV  Details
25 Portfolios Formed on Book-to-Market and Operating Profitability (5 x 5) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Book-to-Market and Investment (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Book-to-Market and Investment (5 x 5) [ex. Dividends]  TXT  CSV  Details
25 Portfolios Formed on Book-to-Market and Investment (5 x 5) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Operating Profitability and Investment (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Operating Profitability and Investment (5 x 5) [ex. Dividends]  TXT  CSV  Details
25 Portfolios Formed on Operating Profitability and Investment (5 x 5) [Daily]  TXT  CSV  Details

Three-way sorts on Size, B/M, OP, and Inv

32 Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4)  TXT  CSV  Details
32 Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4) [ex. Dividends]  TXT  CSV  Details

32 Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4)  TXT  CSV  Details
32 Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4) [ex. Dividends]  TXT  CSV  Details

32 Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4)  TXT  CSV  Details
32 Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4) [ex. Dividends]  TXT  CSV  Details

Univariate sorts on E/P, CF/P, and D/P

Portfolios Formed on Earnings/Price  TXT  CSV  Details
Portfolios Formed on Earnings/Price [ex. Dividends]  TXT  CSV  Details

Portfolios Formed on Cashflow/Price  TXT  CSV  Details
Portfolios Formed on Cashflow/Price [ex. Dividends]  TXT  CSV  Details

Portfolios Formed on Dividend Yield  TXT  CSV  Details
Portfolios Formed on Dividend Yield [ex. Dividends]  TXT  CSV  Details

Bivariate sorts on Size, E/P, CF/P, and D/P

6 Portfolios Formed on Size and Earnings/Price  TXT  CSV  Details
6 Portfolios Formed on Size and Earnings/Price [ex. Dividends]  TXT  CSV  Details

6 Portfolios Formed on Size and Cashflow/Price  TXT  CSV  Details
6 Portfolios Formed on Size and Cashflow/Price [ex. Dividends]  TXT  CSV  Details

6 Portfolios Formed on Size and Dividend Yield  TXT  CSV  Details
6 Portfolios Formed on Size and Dividend Yield [ex. Dividends]  TXT  CSV  Details

Sorts involving Prior Returns

Momentum Factor (Mom)  TXT  CSV  Details
Momentum Factor (Mom) [Daily]  TXT  CSV  Details

6 Portfolios Formed on Size and Momentum (2 x 3)  TXT  CSV  Details
6 Portfolios Formed on Size and Momentum (2 x 3) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Size and Momentum (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Size and Momentum (5 x 5) [Daily]  TXT  CSV  Details

10 Portfolios Formed on Momentum  TXT  CSV  Details
10 Portfolios Formed on Momentum [Daily]  TXT  CSV  Details

Short-Term Reversal Factor (ST Rev)  TXT  CSV  Details
Short-Term Reversal Factor (ST Rev) [Daily]  TXT  CSV  Details

6 Portfolios Formed on Size and Short-Term Reversal (2 x 3)  TXT  CSV  Details
6 Portfolios Formed on Size and Short-Term Reversal (2 x 3) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Size and Short-Term Reversal (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Size and Short-Term Reversal (5 x 5) [Daily]  TXT  CSV  Details

10 Portfolios Formed on Short-Term Reversal  TXT  CSV  Details
10 Portfolios Formed on Short-Term Reversal [Daily]  TXT  CSV  Details

Long-Term Reversal Factor (LT Rev)  TXT  CSV  Details
Long-Term Reversal Factor (LT Rev) [Daily]  TXT  CSV  Details

6 Portfolios Formed on Size and Long-Term Reversal (2 x 3)  TXT  CSV  Details
6 Portfolios Formed on Size and Long-Term Reversal (2 x 3) [Daily]  TXT  CSV  Details

25 Portfolios Formed on Size and Long-Term Reversal (5 x 5)  TXT  CSV  Details
25 Portfolios Formed on Size and Long-Term Reversal (5 x 5) [Daily]  TXT  CSV  Details

10 Portfolios Formed on Long-Term Reversal  TXT  CSV  Details
10 Portfolios Formed on Long-Term Reversal [Daily]  TXT  CSV  Details

Sorts involving Accruals, Market Beta, Net Share Issues, Daily Variance, and Daily Residual Variance

Portfolios Formed on Accruals  TXT  CSV  Details
25 Portfolios Formed on Size and Accruals  TXT  CSV  Details

Portfolios Formed on Market Beta  TXT  CSV  Details
25 Portfolios Formed on Size and Market Beta  TXT  CSV  Details

Portfolios Formed on Net Share Issues  TXT  CSV  Details
25 Portfolios Formed on Size and Net Share Issues  TXT  CSV  Details

Portfolios Formed on Variance  TXT  CSV  Details
25 Portfolios Formed on Size and Variance  TXT  CSV  Details

Portfolios Formed on Residual Variance  TXT  CSV  Details
25 Portfolios Formed on Size and Residual Variance  TXT  CSV  Details

Industry Portfolios

5 Industry Portfolios  TXT  CSV  Details
5 Industry Portfolios [ex. Dividends]  TXT  CSV  Details
5 Industry Portfolios [Daily]  TXT  CSV  Details

10 Industry Portfolios  TXT  CSV  Details
10 Industry Portfolios [ex. Dividends]  TXT  CSV  Details
10 Industry Portfolios [Daily]  TXT  CSV  Details

12 Industry Portfolios  TXT  CSV  Details
12 Industry Portfolios [ex. Dividends]  TXT  CSV  Details
12 Industry Portfolios [Daily]  TXT  CSV  Details

17 Industry Portfolios  TXT  CSV  Details
17 Industry Portfolios [ex. Dividends]  TXT  CSV  Details
17 Industry Portfolios [Daily]  TXT  CSV  Details

30 Industry Portfolios  TXT  CSV  Details
30 Industry Portfolios [ex. Dividends]  TXT  CSV  Details
30 Industry Portfolios [Daily]  TXT  CSV  Details

38 Industry Portfolios  TXT  CSV  Details
38 Industry Portfolios [ex. Dividends]  TXT  CSV  Details
38 Industry Portfolios [Daily]  TXT  CSV  Details

48 Industry Portfolios  TXT  CSV  Details
48 Industry Portfolios [ex. Dividends]  TXT  CSV  Details
48 Industry Portfolios [Daily]  TXT  CSV  Details

49 Industry Portfolios  TXT  CSV  Details
49 Industry Portfolios [ex. Dividends]  TXT  CSV  Details
49 Industry Portfolios [Daily]  TXT  CSV  Details

'''
ds = web.DataReader('49_Industry_Portfolios', 'famafrench')
'''
  0 : Average Value Weighted Returns -- Monthly (59 rows x 10 cols)
  1 : Average Equal Weighted Returns -- Monthly (59 rows x 10 cols)
  2 : Average Value Weighted Returns -- Annual (4 rows x 10 cols)
  3 : Average Equal Weighted Returns -- Annual (4 rows x 10 cols)
  4 : Number of Firms in Portfolios (59 rows x 10 cols)
  5 : Average Firm Size (59 rows x 10 cols)
  6 : Sum of BE / Sum of ME (5 rows x 10 cols)
  7 : Value-Weighted Average of BE/ME (5 rows x 10 cols)
'''
#print(ds['DESCR'])
print(ds[4].head())

# if you wanted to compare the Gross Domestic Products per capita in constant dollars in North America, you would use the search function:
matches = wb.search('gdp.*capita.*const') # web scraping to search for a specific piece of text on Wikipedia pages. STARTING WITH GDP, capita and const need to be in there
# Then you would use the download function to acquire the data from the World Bank’s servers:
print(matches)
dat = wb.download(indicator='NY.GDP.PCAP.KD', country=['US', 'CA', 'MX'], start=2005, end=2008)
print(dat)
# Now imagine you want to compare GDP to the share of people with cellphone contracts around the world.
print(wb.search('cell.*%').iloc[:,:2])
# GET COUNTRY, GDP AND coverage of the mobile network operator's LTE network, measured in percentage of the population.
ind = ['NY.GDP.PCAP.KD', 'IT.MOB.COV.ZS']
dat = wb.download(indicator=ind, country='all', start=2011, end=2011).dropna()
dat.columns = ['gdp', 'cellphone']
print(dat.tail())
# Topic (2 digits)
# General Subject (3 digits)
# Specific Subject (4 digits)
# Extensions (2 digits each)

#mod = smf.ols('cellphone ~ np.log(gdp)', dat).fit()
#print(mod.summary())

rate=pdr.DataReader('DGS10', 'fred') #10-year U.S. Treasury yield 
print(rate)
gdp = web.DataReader('GDP', 'fred')
print(gdp)
inflation = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred')
print(inflation)

# Fetch CPI data from FRED
cpi = web.DataReader('CPIAUCSL', 'fred')
# Calculate the inflation rate by taking the percentage change between the current month's CPI and the CPI from one year ago
inflation_rate = (cpi.iloc[-1] - cpi.iloc[-13]) / cpi.iloc[-13] * 100
#print("Inflation Rate: {:.2f}%".format(inflation_rate))

#print(web.DataReader('ticker=RGDPUS', 'econdb').head())
#print(web.DataReader('dataset=NAMQ_10_GDP&v=Geopolitical entity (reporting)&h=TIME&from=2018-05-01&to=2021-01-01&GEO=[AL,AT,BE,BA,BG,HR,CY,CZ,DK,EE,EA19,FI,FR,DE,EL,HU,IS,IE,IT,XK,LV,LT,LU,MT,ME,NL,MK,NO,PL,PT,RO,RS,SK,SI,ES,SE,CH,TR,UK]&NA_ITEM=[B1GQ]&S_ADJ=[SCA]&UNIT=[CLV10_MNAC]', 'econdb').columns)
print(yf.download(['AAPL', 'V'], start = '2023-01-01', end = '2024-01-01'))