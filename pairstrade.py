import statsmodels.api as sm
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from  statsmodels.tsa.stattools import adfuller
import warnings

('CVX', 'MRO')
('AAPL', 'MSFT')
date_today = dt.date.today()
stock1 = yf.download('CVX', start=(date_today - dt.timedelta(days=1825)), end=date_today)
stock2 = yf.download('MRO', start=(date_today - dt.timedelta(days=1825)), end=date_today)

#Charting relative prices
stock1_close_relative = stock1['Close']/stock1['Close'][0] * 100
stock2_close_relative = stock2['Close']/stock2['Close'][0] * 100
plt.plot(stock1_close_relative, label='CVX')
plt.plot(stock2_close_relative, label='MRO')
plt.xlabel('Time')
plt.ylabel('Relative Close Price')
plt.legend
plt.show()

#Regression Model
y = np.log(stock2['Close'])
x = np.log(stock1['Close'])
x = sm.add_constant(x)
model = sm.OLS(y,x)
results = model.fit()
results.params

#Get spread
alpha = results.params.values[0]
beta = results.params.values[1]
errors = y - (alpha+x['Close']*beta)

#Charting spread
errors.plot(label = "x = CVX; y = MRO \n MRO - CVX")
plt.title(f"Residuals from regression (spread) \n Spread = MRO - ({alpha} + {beta}*CVX)", fontsize=10)
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()

#Test cointegration 
#Dickey Fuller Test
dftest=adfuller(errors, maxlag=1)
dfoutput=pd.Series(dftest[0:4],
                   index=['Test Statistic', "p-value", '#Lags Used', 'Number of Observations Used'],
                   )
critical_values=pd.Series(dftest[4].values(), index=dftest[4].keys())

print(f'Dickey Fuller Result: \n {dfoutput} \n\nDickey Fuller Critical Values:\n {critical_values}')

#z-score
spread = errors
zscore = (spread - np.mean(spread)) / np.std(spread)
zscore.plot(label = "z-score")
plt.title(f"z-score MRO - CVX")
plt.xlabel('Time')
plt.ylabel('Values')
plt.axhline(y=1.2, color='b', label='1.2 threshold')
plt.axhline(y=-1.2, color='b', label='-1.2 threshold')
plt.legend()
plt.show()

#Short stocks
btest = pd.DataFrame()
btest["stock2"] = stock2['Close']
btest["stock1"] = stock1['Close']
btest["Short signal"] = (zscore > 1.2) & (zscore.shift(1) < 1.2)
btest["Short exit"] = (zscore < 1.2) & (zscore.shift(1) > 1.2)
btest["Long signal"] = (zscore < -1.2) & (zscore.shift(1) > -1.2)
btest["Long exit"] = (zscore > -1.2) & (zscore.shift(1) < -1.2)

# Calculate moving averages for stock prices
window_short = 20  # Short-term moving average window
window_long = 50    # Long-term moving average window
std_multiplier = 2  # Multiplier for standard deviation
btest['SMA short 1'] = stock1['Close'].rolling(window=window_short).mean()
btest['Short Std 1'] = stock1['Close'].rolling(window=window_short).std()
btest['SMA long 1'] = stock1['Close'].rolling(window=window_long).mean()
btest['Long Std 1'] = stock1['Close'].rolling(window=window_long).std()

btest['SMA short 2'] = stock2['Close'].rolling(window=window_short).mean()
btest['Short Std 2'] = stock2['Close'].rolling(window=window_short).std()
btest['SMA long 2'] = stock2['Close'].rolling(window=window_long).mean()
btest['Long Std 2'] = stock1['Close'].rolling(window=window_long).std()

# Calculate upper and lower Bollinger Bands
btest['UpperBand'] = btest['SMA short 1'] + std_multiplier * btest['Short Std 1']
btest['LowerBand'] = btest['SMA short 1'] - std_multiplier * btest['Short Std 1']
btest['UpperBand'] = btest['SMA short 2'] + std_multiplier * btest['Short Std 1']
btest['LowerBand'] = btest['SMA long 2'] - std_multiplier * btest['Short Std 1']

btest['Stock1_above_upper_band'] = (stock1['Close'] > stock1['UpperBand']).astype(int)
btest['Stock1_below_lower_band'] = (stock1['Close'] < stock1['LowerBand']).astype(int)
btest['Stock2_above_upper_band'] = (stock2['Close'] > stock2['UpperBand']).astype(int)
btest['Stock2_below_lower_band'] = (stock2['Close'] < stock2['LowerBand']).astype(int)
 
print(btest)
# Introduce stop-loss levels
stop_loss_threshold = 0.10  # 10% stop-loss
stop_loss_triggered = False

spread_side=None; counter=-1
backtest_result = []; indicator = 0
for time, signals_stock in btest.iterrows():
    counter+=1
    stock2_, stock1_, short_signal, short_exit, long_signal, long_exit = signals_stock

    if spread_side == None:
        return_stock2=0
        return_stock1=0
        backtest_result.append([time,return_stock2,return_stock1,spread_side])

        if short_signal == True:
            spread_side = 'short'
        elif long_signal == True:
            spread_side = 'long'
    
    elif spread_side == 'long':
        return_stock2 = btest['stock2'][counter] / btest['stock2'][counter-1] -1.
        return_stock1 = btest['stock1'][counter] / btest['stock1'][counter-1] -1.
        backtest_result.append([time,return_stock2, -return_stock1,spread_side])

        if long_exit == True:
            spread_side = None
    
    elif spread_side == 'short':
        return_stock2 = btest['stock2'][counter] / btest['stock2'][counter-1] -1.
        return_stock1 = btest['stock1'][counter] / btest['stock1'][counter-1] -1.
        backtest_result.append([time,-return_stock2, return_stock1,spread_side])

        if short_exit == True:
            spread_side = None

backtest_pandas = pd.DataFrame(backtest_result)
backtest_pandas.columns = ['Date', 'stock2', 'stock1', 'Side']
backtest_pandas['stocks2 PL'] = np.cumprod(backtest_pandas['stock2'] + 1.)
backtest_pandas['stocks1 PL'] = np.cumprod(backtest_pandas['stock1'] + 1.)
backtest_pandas['Total PL'] = (backtest_pandas['stocks1 PL'] + backtest_pandas['stocks2 PL'] ) / 2
#print(btest, backtest_pandas, backtest_result)
backtest_pandas[['Total PL']].plot(label = 'Evolution Profit and Loss')
plt.title('Equity Curve Pairs Trading')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()
final_return = (backtest_pandas['Total PL'].iloc[-1] - 1)
final_return_ptg = final_return * 100
print('Final Returns: ', final_return_ptg, '%')