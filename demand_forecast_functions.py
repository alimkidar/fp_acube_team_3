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

    def update_db(self):
        try:
            df_forecast = pd.DataFrame(self.df_forecast)
            df_forecast['Forecast'] = df_forecast['Forecast'].apply(lambda x: int(x))
            df_forecast.to_gbq('alim_hanif.tab_forecast',if_exists ='append', project_id='minerva-da-coe')
            return print(f'DEBUG:Update-BQ-Forecast')
        except:
            return print("ERROR:FORECAST-UPDATE_DB")        
