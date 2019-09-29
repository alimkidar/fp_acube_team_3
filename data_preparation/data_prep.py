import pandas as pd
import numpy as np
import re
import textdistance

from data_prep_functions import *
from collections import Counter

def preprocessing(df_all):
    df_all = df_all.apply(lambda x: x.astype(str).str.lower())
    # get tokens
    data_token = get_token(df_all, 'product_name')
    # get bag of words by count
    bow = pd.DataFrame.from_dict(Counter(data_token), orient='index').reset_index() 
    return df_all, data_token, bow

def normalize_xiaomi(df_all, data_token, bow):
    print("DEBUG: normalize_xiaomi-get list of error")
    list_xiaomi = text_distance_mra(bow, 'index', 'xiaomi', 0.4)
    list_redmi = text_distance_mra(bow, 'index', 'redmi', 0.05)
    print("DEBUG: normalize_xiaomi-autocorrect xiaomi")
    df_all['product_name'] = df_all.apply(lambda x: ganti(x['product_name'], list_xiaomi, 'xiaomi'), axis=1)

    df_all['product_name'] = df_all.apply(lambda x: ganti(x['product_name'], list_redmi, 'redmi'), axis=1)

    # string matching for brand
    brands = ['xiaomi', 'redmi']
    for brand in brands:
        df_all[f"is_{brand}"] = np.vectorize(lambda x: 1 if brand in x else 0)(df_all['product_name'])

    # combine xiaomi and redmi
    df_all['is_xiaomi'] = np.vectorize(lambda x, y: 1 if x == 1 or y == 1 else 0)(df_all['is_xiaomi'], df_all['is_redmi'])

    df_all = df_all.drop('is_redmi', axis=1)

    print("DEBUG: normalize_xiaomi-regex")
    df_xiaomi = df_all[df_all['is_xiaomi'] == 1]

    def normalize(product_name):
        product_name = product_name.replace('xiaomi', ' xiaomi ')
        product_name = product_name.replace('redmi', ' redmi ')
        if 'xiaomi ' not in product_name:
            product_name = 'xiaomi ' + product_name
        else:
            pass
        product_name = product_name.replace('xiaomi xiaomi', ' xiaomi ')
        return product_name
    df_xiaomi['product_name'] = normalize(df_xiaomi['product_name'])
    df_xiaomi['product_name'] = np.vectorize(lambda x: re.sub(f"(xiaomi)?.*(redmi)",'xiaomi redmi',x))(df_xiaomi['product_name'])
    df_xiaomi['type'] = np.vectorize(find_brand)(r"(xiaomi[\s+\w+]*?\d)(\w*)", df_xiaomi['product_name'])
 
    return df_xiaomi

def normalize_iphone(df_all, data_token, bow):
    print("DEBUG: normalize_iphone-get list of error")
    df_all['product_name'] = np.vectorize(lambda x: x.replace('iphone', 'iphone '))(df_all['product_name'])
    list_iphone = text_distance_mra(bow, 'index', 'iphone', 0.45)
    print("DEBUG: normalize_iphone-autocorrect iphone")
    df_all['product_name'] = df_all.apply(lambda x: ganti(x['product_name'], list_iphone, 'iphone'), axis=1)

    # string matching for brand
    brands = ['iphone']
    for brand in brands:
        df_all[f"is_{brand}"] = np.vectorize(lambda x: 1 if brand in x else 0)(df_all['product_name'])

    print("DEBUG: normalize_iphone-regex")
    def normalize(product_name):
        product_name = product_name.replace('iphone', ' iphone ')
        product_name = product_name.replace('iphone iphone', 'iphone')
        product_name = product_name.replace('iphone iphone', 'iphone')
        product_name = product_name.replace('+','+ ')
        return product_name
    df_iphone = df_all[df_all['is_iphone'] == 1]
    

    df_iphone['product_name'] = normalize(df_iphone['product_name'])
    df_iphone['product_name'] = np.vectorize(lambda x: re.sub(r"(\d+gb|\d+\sgb)",' ',str(x)))(df_iphone['product_name'])
    df_iphone['type'] = np.vectorize(find_brand)(r"(iphone[\s+\w+]*?[\d|x]\+?)(\w*)(\s)?(pro|lite|plus|x|max)?", df_iphone['product_name'])

    return df_iphone

def normalize_samsung(df_all, data_token, bow):
    print("DEBUG: normalize_samsung-get list of error")
    def adhoc_normalize(product_name):
        product_name = product_name.replace('samsung', 'samsung ')
        product_name = product_name.replace('galaxy', 'galaxy ')
        product_name = product_name.replace('note', 'note ')
        return product_name
    df_all['product_name'] = np.vectorize(adhoc_normalize)(df_all['product_name'])


    list_samsung = text_distance_mra(bow, 'index', 'samsung', 0.21)
    print("DEBUG: normalize_samsung-autocorrect iphone")
    df_all['product_name'] = df_all.apply(lambda x: ganti(x['product_name'], list_samsung, 'samsung'), axis=1)


    # string matching for brand
    brands = ['samsung', 'galaxy']
    for brand in brands:
        df_all[f"is_{brand}"] = np.vectorize(lambda x: 1 if brand in x else 0)(df_all['product_name'])

    # combine samsung and galaxy
    df_all['is_xiaomi'] = np.vectorize(lambda x, y: 1 if x == 1 or y == 1 else 0)(df_all['is_samsung'], df_all['is_galaxy'])
    df_all = df_all.drop('is_galaxy', axis=1)

    print("DEBUG: normalize_samsung-regex")
    df_samsung = df_all[df_all['is_samsung'] == 1]
    def normalize(product_name):
        product_name = product_name.replace('samsung', ' '+'samsung'+' ')
        product_name = product_name.replace('samsung samsung', 'samsung')
        product_name = product_name.replace('+','+ ')
        return product_name
    
    df_samsung['product_name'] = normalize(df_samsung['product_name'])
    df_samsung['product_name'] = np.vectorize(lambda x: re.sub(r"(\d+gb|\d+\sgb)",' ',str(x)))(df_samsung['product_name'])
    df_samsung['product_name'] = np.vectorize(lambda x: re.sub(r"\s+",' ',str(x)))(df_samsung['product_name'])
    df_samsung['type'] = np.vectorize(find_brand)(r"(samsung[\s+\w+]*?[\d]\+?)(\w*)(\s)?(pro|lite|plus|max|prime|duos|ace|edge)?(\s)?(pro|lite|plus|max|prime|duos|ace|edge)?", df_samsung['product_name'])
     
    return df_samsung

def normalize_oppo(df_all, data_token, bow):
    print("DEBUG: normalize_oppo-get list of error")
    df_all['product_name'] = df_all['product_name'].apply(lambda x: x.replace('oppo', 'oppo '))
    list_oppo = text_distance_mra(bow, 'index', 'oppo', 0.5) #regex_ipyb
    print("DEBUG: normalize_oppo-autocorrect oppo")
    df_all['product_name'] = df_all.apply(lambda x: ganti(x['product_name'], list_oppo, 'oppo'), axis=1)

    # string matching for brand
    brands = ['oppo']
    for brand in brands:
        df_all[f"is_{brand}"] = np.vectorize(lambda x: 1 if brand in x else 0)(df_all['product_name'])

    print("DEBUG: normalize_oppo-regex")
    df_oppo = df_all[df_all['is_oppo'] == 1]
    def normalize(product_name):
        product_name = product_name.replace('oppo', ' '+'oppo'+' ')
        product_name = product_name.replace('realme', ' '+'oppo realme'+' ') 
        product_name = product_name.replace('oppo oppo', 'oppo')
        product_name = product_name.replace('+','+ ')
        return product_name

    df_oppo['product_name'] = np.vectorize(normalize)(df_oppo['product_name'])
 
    df_oppo['product_name'] = normalize(df_oppo['product_name'])
    df_oppo['product_name'] = np.vectorize(lambda x: re.sub(r"(\d+gb|\d+\sgb)",' ',str(x)))(df_oppo['product_name'])
    df_oppo['product_name'] = np.vectorize(lambda x: re.sub(r"(grs|garansi|resmi|termurah|murah|ram).*",' ',str(x)))(df_oppo['product_name'])
    df_oppo['type'] = np.vectorize(find_brand)(r"(oppo[\s+\w+]*?[\d]\+?)(\w*)(\s)?(pro|lite|plus|max|prime|duos|ace|edge)?", df_oppo['product_name'])
 
    return df_oppo
def get_brand(df, col_name, min_dp):
    df_result = df.groupby(col_name).count()
    cols = df_result.columns
    df_result = df_result[df_result[cols[0]]> min_dp]
    return df_result
def main():
    # data_list_product_name consists of list of products
    # read data
    
    print("DEBUG: Pre-processing Data")
    df_all = pd.read_pickle('data_list_product_name.pkl')
    df_all, data_token, bow = preprocessing(df_all)

    print("DEBUG: Normalizing Xiaomi Data")
    df_xiaomi = normalize_xiaomi(df_all, data_token, bow)
    df_brand_xiaomi = get_brand(df_xiaomi, 'type', 10)


    print("DEBUG: Normalizing Iphone Data")
    df_iphone = normalize_iphone(df_all, data_token, bow)
    df_brand_iphone = get_brand(df_iphone, 'type', 2)


    print("DEBUG: Normalizing Samsung Data")
    df_samsung = normalize_samsung(df_all, data_token, bow)
    df_brand_samsung = get_brand(df_samsung, 'type', 20)


    print("DEBUG: Normalizing Oppo Data")
    df_oppo = normalize_samsung(df_all, data_token, bow)
    df_brand_oppo = get_brand(df_oppo, 'type', 5)

    df_xiaomi.to_csv('df_xiaomi.csv')
    df_iphone.to_csv('df_iphone.csv')
    df_samsung.to_csv('df_samsung.csv')
    df_oppo.to_csv('df_oppo.csv')
    return df_xiaomi, df_iphone, df_samsung, df_oppo
main()
