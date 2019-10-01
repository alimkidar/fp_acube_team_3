from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SARIMAX
from google.cloud import bigquery
from datetime import date, timedelta
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

warnings.filterwarnings("ignore")
def get_data(df, cluster, warehouse):
    try:
        df = df[(df['Cluster'] == cluster)&(df['Warehouse'] == warehouse)]
        df = df.sort_values('Week').set_index('Week')
        df = df.reset_index()
        return df
    except:
        return print(f"ERROR:GET-DATA:{cluster}:{warehouse}")
def generate_attrib(df):
    try:
        df["DT"] = pd.to_datetime(df.Week.astype(str)+ df.Year.astype(str).add('-1') ,format='%W%Y-%w') + timedelta(weeks=1)
        df['W_F'] = df['DT'].apply(lambda x: int(x.strftime("%W")) )
        df['Y_F'] = df['DT'].apply(lambda x: int(x.strftime("%Y")) )
        WF = df.tail(1).W_F.values[0]
        YF = df.tail(1).Y_F.values[0]
        Cluster = df.tail(1).Cluster.values[0]
        Warehouse = df.tail(1).Warehouse.values[0]
        return Cluster, Warehouse, WF, YF
    except:
        return print(f"ERROR:GENERATE_ATTRIB")
class Forecast():
    def __init__(self):
        try:
            self.df = pd.read_gbq("""SELECT * FROM alim_hanif.tab_actual""", project_id='minerva-da-coe', dialect='standard')
            self.df_forecast = []
            print("SUCCESS:FORECAST-INIT")
        except:
            print('ERROR:FORECAST-INIT')

    def MA_f(self, df, p):
        try:
            forecast = df['Actual'].tail(p).mean()

            Cluster, Warehouse, WF, YF = generate_attrib(df)
            self.df_forecast.append({'Cluster':Cluster, 'Warehouse':Warehouse, 'Year':YF, "Week": WF, "Forecast":forecast})
            return print(f'DEBUG:Forecast:{Cluster}:{Warehouse}:{YF}:{WF}:{forecast}')
        except:
            return print(f"ERROR:FORECAST-MA")
    def SES_f(self, df, a):
        try:
            simpleexp = SimpleExpSmoothing(np.array(np.array(df['Actual'])))
            fit_simpleexp = simpleexp.fit(smoothing_level=a,optimized=False)
            forecast = fit_simpleexp.forecast()[0]

            Cluster, Warehouse, WF, YF = generate_attrib(df)
            self.df_forecast.append({'Cluster':Cluster, 'Warehouse':Warehouse, 'Year':YF, "Week": WF, "Forecast":forecast})
            return print(f'DEBUG:Forecast:{Cluster}:{Warehouse}:{YF}:{WF}:{forecast}')
        except:
            return print("ERROR:FORECAST-SES")
    def DEF_f(self, df, alpha, beta):
        try:
            double_exp = Holt(np.array(df['Actual']), exponential=True)
            fit_double_exp = double_exp.fit(smoothing_level=alpha, smoothing_slope=beta,optimized=False)
            forecast = fit_double_exp.forecast()[0]

            Cluster, Warehouse, WF, YF = generate_attrib(df)
            self.df_forecast.append({'Cluster':Cluster, 'Warehouse':Warehouse, 'Year':YF, "Week": WF, "Forecast":forecast})
            return print(f'DEBUG:Forecast:{Cluster}:{Warehouse}:{YF}:{WF}:{forecast}')
        except:
            return print("ERROR:FORECAST-DEF")
    def DEF_damping_f(self, df, alpha, beta):
        try:
            double_exp = Holt(np.array(df['Actual']), exponential=True, damped=True)
            fit_double_exp = double_exp.fit(smoothing_level=alpha, smoothing_slope=beta,optimized=False)
            forecast = fit_double_exp.forecast()[0]

            Cluster, Warehouse, WF, YF = generate_attrib(df)
            self.df_forecast.append({'Cluster':Cluster, 'Warehouse':Warehouse, 'Year':YF, "Week": WF, "Forecast":forecast})
            return print(f'DEBUG:Forecast:{Cluster}:{Warehouse}:{YF}:{WF}:{forecast}')
        except:
            return print("ERROR:FORECAST-DEF_DAMPING")
    def holt_winter_f(self, df, trend, seasonal):
        try:
            holt_w = ExponentialSmoothing(np.array(df['Actual']), trend=trend, seasonal=seasonal, seasonal_periods=4)
            fit_holt_w = holt_w.fit(use_boxcox=True, disp=0)
            forecast = fit_holt_w.forecast()[0]

            Cluster, Warehouse, WF, YF = generate_attrib(df)
            self.df_forecast.append({'Cluster':Cluster, 'Warehouse':Warehouse, 'Year':YF, "Week": WF, "Forecast":forecast})
            return print(f'DEBUG:Forecast:{Cluster}:{Warehouse}:{YF}:{WF}:{forecast}')
        except:
            return print("ERROR:FORECAST-HOLT_WINTER")
    def ARIMA_f(self, df, p, d, q, boxcox=False):
        try:
            arima_mod = ARIMA(np.array(df['Actual']), order=[p,d,q])
            fit_arima = arima_mod.fit(use_boxcox=boxcox, disp=0)
            forecast = fit_arima.forecast()[0][0]

            Cluster, Warehouse, WF, YF = generate_attrib(df)
            self.df_forecast.append({'Cluster':Cluster, 'Warehouse':Warehouse, 'Year':YF, "Week": WF, "Forecast":forecast})
            return print(f'DEBUG:Forecast:{Cluster}:{Warehouse}:{YF}:{WF}:{forecast}')
        except:
            return print("ERROR:FORECAST-ARIMA")

    def SARIMA_f(self, df, pdq, s):
        try:
            sarima_mod = SARIMAX(np.array(df['Actual']), order=pdq, seasonal_order=s)
            fit_sarima = sarima_mod.fit(use_boxcox=True, disp=0)
            forecast = fit_sarima.forecast()[0]

            Cluster, Warehouse, WF, YF = generate_attrib(df)
            self.df_forecast.append({'Cluster':Cluster, 'Warehouse':Warehouse, 'Year':YF, "Week": WF, "Forecast":forecast})
            return print(f'DEBUG:Forecast:{Cluster}:{Warehouse}:{YF}:{WF}:{forecast}')
        except:
            return print("ERROR:FORECAST-SARIMA")

    def create_dataset(self, dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    def lstm(self, df):
        try:
            dataset = df.Actual.values #numpy.ndarray
            dataset = dataset.astype('float32')
            dataset = np.reshape(dataset, (-1, 1))
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            train_size = int(len(dataset) * 0.90)
            test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

            look_back = 3
            X_train, Y_train = create_dataset(train, look_back)
            X_test, Y_test = create_dataset(test, look_back)

            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))  

            model = Sequential()
            model.add(LSTM(100, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.5))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

            history = model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_test, Y_test), 
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

            train_predict = model.predict(X_train)
            test_predict = model.predict(X_test)
            
            train_predict = scaler.inverse_transform(train_predict)
            Y_train = scaler.inverse_transform([Y_train])
            test_predict = scaler.inverse_transform(test_predict)
            Y_test = scaler.inverse_transform([Y_test])

            forecast = test_predict

            Cluster, Warehouse, WF, YF = generate_attrib(df)
            self.df_forecast.append({'Cluster':Cluster, 'Warehouse':Warehouse, 'Year':YF, "Week": WF, "Forecast":forecast})

            return print(f'DEBUG:Forecast:{Cluster}:{Warehouse}:{YF}:{WF}:{forecast}')
        except:
            return print("ERROR:FORECAST-LSTM")

    def update_db(self):
        try:
            df_forecast = pd.DataFrame(self.df_forecast)
            df_forecast['Forecast'] = df_forecast['Forecast'].apply(lambda x: int(x))
            df_forecast.to_gbq('alim_hanif.tab_forecast',if_exists ='append', project_id='minerva-da-coe')
            return print(f'DEBUG:Update-BQ-Forecast')
        except:
            return print("ERROR:FORECAST-UPDATE_DB")        

    