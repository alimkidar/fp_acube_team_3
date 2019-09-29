from google.cloud import bigquery
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

clusters = [0,1,3,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
warehouses = [1,2,3]
regions = ['Jabodetabek','Jawa Barat', 'Jawa Timur']
combinations = []

def get_combination(clusters, warehouses, regions):
    for cluster in clusters:
        for warehouse in warehouses:
            for region in regions:
                if (warehouse == 1 and region == 'Jabodetabek') or (warehouse == 2 and region == 'Jawa Barat') or (warehouse == 3 and region == 'Jawa Timur'):
                    combinations.append({'cluster':cluster, 'warehouse':warehouse, 'region':region})
                else:
                    pass      
    df_comb = pd.DataFrame(combinations)
    return df_comb

def query_request_actual(df_comb):
    df_all = pd.DataFrame()
    count = 1
    for index, row in df_comb.iterrows():
        cluster = str(row['cluster'])
        region = str(row['region'])
        warehouse = row['warehouse']
        query=f"""
            SELECT
            cluster,
            warehouse,
            payment_verified_wib_year as Year,
            CAST(week_ AS int64) as Week,
            SUM(quantity) as Actual
            FROM
            alim_hanif.data1_fo_result
            WHERE
            cluster = {cluster}
            AND warehouse = '{warehouse}'
            AND new_origin_city != '{region}'
            Group BY week_, cluster, warehouse, payment_verified_wib_year
        """
    df_query = pd.read_gbq(query, project_id='minerva-da-coe', dialect='standard')
    df_all = df_all.append(df_query)
    print(f'DEBUG:Query Requests-Actual:{str(count)}:{cluster}:{region}:{warehouse}:INFO-{str(len(df_query))}')
    df_query = ''
    count += 1
    return df_all

def query_update_actual(df_all, if_exists='fail'):
    df_all.columns = ['Cluster', 'Warehouse', 'Year', 'Week', 'Actual']
    df_all = df_all.apply(pd.to_numeric)
    df_all.to_gbq('alim_hanif.tab_actual',if_exists =if_exists, project_id='minerva-da-coe')
    print(f'DEBUG:Query Update-Actual-Success')
def main():
    df_comb = get_combination(clusters, warehouses, regions)
    df_all = query_request_actual(df_comb)
    query_update_actual(df_all, 'replace')
main()