from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SARIMAX
from google.cloud import bigquery
import pandas as pd
import numpy as np
import warnings
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from keras.callbacks import History
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")

def init():
    df = pd.read_gbq("""SELECT * FROM alim_hanif.tab_actual""", project_id='minerva-da-coe', dialect='standard')
    return df

def get_data(df, cluster, warehouse):
    df = df[(df['Cluster'] == cluster)&(df['Warehouse'] == warehouse)]
    df = df.sort_values('Week').set_index('Week')
    return df

def MA_f(df, p):
    return df['Actual'].tail(p).mean()
def SES_f(df, a):
    simpleexp = SimpleExpSmoothing(np.array(np.array(df['Actual'])))
    fit_simpleexp = simpleexp.fit(smoothing_level=a,optimized=False)
    forecast = fit_simpleexp.forecast()
    return forecast[0]
def DEF_f(df, alpha, beta):
    double_exp = Holt(np.array(df['Actual']), exponential=True)
    fit_double_exp = double_exp.fit(smoothing_level=alpha, smoothing_slope=beta,optimized=False)
    forecast = fit_double_exp.forecast()
    return forecast[0]
def DEF_damping_f(df, alpha, beta):
    double_exp = Holt(np.array(df['Actual']), exponential=True, damped=True)
    fit_double_exp = double_exp.fit(smoothing_level=alpha, smoothing_slope=beta,optimized=False)
    forecast = fit_double_exp.forecast()
    return forecast[0]
def holt_winter_f(df, trend, seasonal):
    holt_w = ExponentialSmoothing(np.array(df['Actual']), trend=trend, seasonal=seasonal, seasonal_periods=4)
    fit_holt_w = holt_w.fit(use_boxcox=True)
    forecast = fit_holt_w.forecast()
    return forecast[0]
def ARIMA_forecast(df, p, d, q):
    arima_mod = ARIMA(np.array(df['Actual']), order=[p,d,q])
    fit_arima = arima_mod.fit(use_boxcox=True)
    forecast = fit_arima.forecast()
    return forecast[0][0]
def SARIMA_f(df, pdq, s):
    sarima_mod = SARIMAX(np.array(df['Actual']), order=pdq, seasonal_order=s)
    fit_sarima = sarima_mod.fit(use_boxcox=True)
    forecast = fit_sarima.forecast()
    return forecast[0]

def lstm(df, model):
    