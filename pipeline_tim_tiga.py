from datetime import timedelta, datetime
import json
from airflow import DAG
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_check_operator import BigQueryCheckOperator

import pandas_gbq
import pydata_google_auth
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators import bigquery_operator
from airflow.operators.postgres_operator import PostgresOperator
from datetime import datetime, timedelta
import psycopg2 as pg
import pandas as pd

from google.cloud import storage
from oauth2client.client import GoogleCredentials
import os

import forecast as fc
import update_actual
import update_forecast

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "<pathtomycredentials>"

default_args = {
    'owner': 'tim_tiga',
    #'depends_on_past': False,
    'start_date': datetime(2019, 8, 23),
    'email': ['rizky.putranto@tokopedia.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    #'retry_delay': timedelta(minutes=1),
}

dag = DAG('final_project_tim_tiga', default_args=default_args)

#1 Connect bq
def bq_connection(ds, **ags):
    SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive',
    ]
    credentials = pydata_google_auth.get_user_credentials(
    SCOPES,
    auth_local_webserver=False)
    print('the connection is available')
    return credentials
    
#2 Get data
def bq_query(ds, **ags):
    #pull the data from previous task
    credentials = ags['task_instance'].xcom_pull(task_ids='bq_connection')
    #conditional: if not exist, select all, else: select just the newest (check to the bq)
    df = update_actual.main_update_actual(credentials)

    return df

#3 Forecast
def model_forecast(ds, **ags):
    df = ags['task_instance'].xcom_pull(task_ids='start_query')
    update_forecast.main()
    return df

#5 Save BQ
def save_bq(ds, **ags):
    # df = ags['task_instance'].xcom_pull(task_ids='forecast')

    print('Data saved')   

t_bq_conn = PythonOperator(
    task_id='bq_connection',
    provide_context=True,
    python_callable=bq_connection,
    xcom_push=True,
    dag=dag)

t_bq_query = PythonOperator(
    task_id='start_query',
    provide_context=True,
    python_callable=bq_query,
    dag=dag)

t_forecast = PythonOperator(
    task_id = 'forecast',
    provide_context=True,
    python_callable=model_forecast,
    dag=dag)

t_save_bq = PythonOperator(
    task_id='save_to_bq',
    provide_context=True,
    python_callable=save_bq,
    dag=dag)

t_bq_conn >> t_bq_query >> t_forecast >> t_save_bq