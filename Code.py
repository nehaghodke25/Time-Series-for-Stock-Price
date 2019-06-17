Time Series- Stock Prices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import math
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import itertools
import warnings
warnings.filterwarnings('ignore')
from nsepy import get_history
from datetime import datetime, date



"""Data Fetching usin NSEPY
Just for an example Here we are Fetching SBIN data"""

data=get_history(symbol="SBIN", series= "EQ", start= date(2015,1,1),end= date(2019,4,23))
ts = data['Close']

"""Data Normalization"""
ts_log = np.log(ts)
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)

"""ARIMA Model"""
model = ARIMA(ts_log_diff, order=(4, 1, 0))
result_ARIMA = model.fit(disp=-1)

"""Training and Testing Data"""
size = int(len(ts_log)-25)
train, test = ts_log[0:size], ts_log[size:len(ts_log)]
history = [x for x in train]
predictions = list()
print("Printing Predicted vs Expected Values")
print('\n')

"""Training the model and Forecasting"""
for t in range(len(test)):
    model = ARIMA(history, order=(4, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

"""Validating The Model"""
error = mean_squared_error(test, predictions)
prediction_series = pd.Series(predictions, index=test.index)
print(error)


"""Plotting Forecasted vs Observated values"""
plt.subplot()
plt.title= ('Predictions vs observed values')
plt.xlabel = ('Date')
plt.ylabel= ('Price')
plt.plot(ts[-100:], 'o', label='Observed')
plt.plot(np.exp(prediction_series), 'g', label='Rolling one step out of sample forecast')
plt.legend(loc='upper left')
plt.show()

forecast = model_fit.forecast(steps=2)
print(np.exp(forecast[0]))


"""Confidence Interval for Forecasting"""
intervals = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0.1, 0.05, 0.01]
for a in intervals:
    forecast, stderr, conf = model_fit.forecast(steps=1,alpha=a)
    print('%.1f%% Confidence Interval: %.3f between %.3f and %.3f' % ((1-a)*100, np.exp(forecast), np.exp(conf[0][0]), np.exp(conf[0][1])))
