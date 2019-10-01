from demand_forecast_functions import Forecast, get_data

import pandas as pd 
import numpy as np 

def main():
    f = Forecast()

    # Subsetting Data

    #Warehouse 1. Jabodetabek
    print(f"DEBUG:Load-WH1")
    w1c0 = get_data(f.df, 0, 1)
    w1c1 = get_data(f.df, 1, 1)
    w1c3 = get_data(f.df, 3, 1)
    w1c4 = get_data(f.df, 4, 1)
    w1c10 = get_data(f.df, 10, 1)
    w1c11 = get_data(f.df, 11, 1)
    w1c12 = get_data(f.df, 12, 1)
    w1c13 = get_data(f.df, 13, 1)
    w1c14 = get_data(f.df, 14, 1)
    w1c15 = get_data(f.df, 15, 1)
    w1c16 = get_data(f.df, 16, 1)
    w1c17 = get_data(f.df, 17, 1)
    w1c18 = get_data(f.df, 18, 1)
    w1c19 = get_data(f.df, 19, 1)
    w1c20 = get_data(f.df, 20, 1)
    w1c21 = get_data(f.df, 21, 1)
    w1c22 = get_data(f.df, 22, 1)
    w1c23 = get_data(f.df, 23, 1)
   

    #Warehouse 2. Bandung
    print(f"DEBUG:Load-WH2")
    w2c0 = get_data(f.df, 0, 2)
    w2c1 = get_data(f.df, 1, 2)
    w2c3 = get_data(f.df, 3, 2)
    w2c4 = get_data(f.df, 4, 2)
    w2c10 = get_data(f.df, 10, 2)
    w3c11 = get_data(f.df, 11, 3)
    w2c12 = get_data(f.df, 12, 2)
    w2c13 = get_data(f.df, 13, 2)
    w2c14 = get_data(f.df, 14, 2)
    w2c15 = get_data(f.df, 15, 2)
    w2c16 = get_data(f.df, 16, 2)
    w2c17 = get_data(f.df, 17, 2)
    w2c18 = get_data(f.df, 18, 2)
    w2c19 = get_data(f.df, 19, 2)
    w2c20 = get_data(f.df, 20, 2)
    w2c21 = get_data(f.df, 21, 2)
    w2c22 = get_data(f.df, 22, 2)
    w2c23 = get_data(f.df, 23, 2)

    #Warehouse 3. Surabaya
    print(f"DEBUG:Load-WH3")
    w3c0 = get_data(f.df, 0, 3)
    w3c1 = get_data(f.df, 1, 3)
    w3c3 = get_data(f.df, 3, 3)
    w3c4 = get_data(f.df, 4, 3)
    w3c10 = get_data(f.df, 10, 3)
    w3c11 = get_data(f.df, 11, 3)
    w3c12 = get_data(f.df, 12, 3)
    w3c13 = get_data(f.df, 13, 3)
    w3c14 = get_data(f.df, 14, 3)
    w3c15 = get_data(f.df, 15, 3)
    w3c16 = get_data(f.df, 16, 3)
    w3c17 = get_data(f.df, 17, 3)
    w3c18 = get_data(f.df, 18, 3)
    w3c19 = get_data(f.df, 19, 3)
    w3c20 = get_data(f.df, 20, 3)
    w3c21 = get_data(f.df, 21, 3)
    w3c22 = get_data(f.df, 22, 3)
    w3c23 = get_data(f.df, 23, 3)

    # Demand Forecasting
    # Warehouse 1
    print(f"DEBUG:Forecast-WH1")
    f.ARIMA_f(w1c0, 0, 0, 2)
    f.SARIMA_f(w1c1, [1,1,1], [1,1,1,4])
    # w1c3 - 
    # f.DEF_damping_f(w1c4, 0.1, 0.7, 0.9) # Need Edit
    f.SARIMA_f(w1c10, [1,0,0,], [0,0,0,4])
    f.ARIMA_f(w1c11, 2,1,2)
    f.DEF_f(w1c12, 0.3, 0.8)
    f.DEF_f(w1c13, 0.3, 0.8)
    # w1c14
    # w1c15
    f.DEF_f(w1c16, 0.2, 0.9)
    f.ARIMA_f(w1c17, 2,0,0)
    f.ARIMA_f(w1c18, 1,0,1)
    f.DEF_f(w1c19, 0.1, 0.5)
    f.DEF_f(w1c20, 0.9, 0.0)
    # f.DEF_damping_f(w1c21, 0.3, 0.6, 0.9)
    f.SARIMA_f(w1c22, [0,0,0], [1,0,0,4])
    f.DEF_f(w1c23, 0.1, 0.7)

    # Warehouse 2
    print(f"DEBUG:Forecast-WH2")
    f.ARIMA_f(w2c0, 1,0,1)
    f.ARIMA_f(w2c1, 0,0,2,boxcox=True)
    f.DEF_f(w2c3, 0.9, 0.0)
    f.DEF_f(w2c4, 0.2, 0.6)
    f.SARIMA_f(w2c10, [0,1,1],[0,1,0,4])
    f.SARIMA_f(w2c11, [0,1,0],[1,1,0,4])
    f.SARIMA_f(w2c12, [0,0,0],[0,1,1,4])
    f.MA_f(w2c13, 15)
    f.SARIMA_f(w2c14, [0,0,1], [0,0,1,4])
    # w2c15
    f.SARIMA_f(w2c16, [0,0,0], [1,1,1,4])
    # w2c17 LSTM
    w2c18
    # w2c19 LSTM
    # w2c20 LSTM
    f.SARIMA_f(w2c21, [1,1,0], [0,1,0,4])
    f.ARIMA_f(w2c22, 0,0,1)
    f.SARIMA_f(w2c23, [1,1,0], [1,1,1,4])


    # Warehouse 3
    print(f"DEBUG:Forecast-WH3")
    f.SARIMA_f(w3c0, [1,1,0], [0,1,1,4])
    f.MA_f(w3c1, 2)
    f.DEF_f(w3c3, 0.9, 0.1)
    # f.DEF_damping_f(w3c4, 0.5, 0.8, 0.9)
    f.SARIMA_f(w3c10, [1,0,1], [1,0,1,4])
    f.SARIMA_f(w3c11, [0,0,0], [1,1,0,4])
    f.SARIMA_f(w3c12, [1,0,0], [0,0,1,4])
    f.SARIMA_f(w3c13, [1,0,1], [0,1,1,4])
    f.DEF_f(w3c14, 0.4, 0.8)
    # w3c15
    # w3c16 LSTM
    f.DEF_f(w3c17, 0.1, 0.5)
    f.MA_f(w3c18, 35)
    f.SARIMA_f(w3c19, [0,0,0],[1,0,0,4])
    w3c20
    f.SARIMA_f(w3c21, [0,0,1],[1,0,1,4])
    f.SES_f(w3c22, 0.22)
    f.ARIMA_f(w3c23, 0,0,2)

    f.update_db()