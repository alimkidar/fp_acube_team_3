import pandas as pd
import numpy as np
import re
import textdistance

# regex for search brand
def find_brand(regex, string):
    text = ''
    x = re.findall(regex, string)
    if len(x) != 0:
        for i in x[0]:
            text += i
        return text
    else:
        return '-'
    
def fix_typo(text, list_typo, correct_string):
    for i in list_typo:
        text = text.replace(i, correct_string)
    return text
    
def ganti(x,list_y,z):
    sentence = ''
    tokens = x.split()
    for token in tokens:
        if token in list_y:
            sentence += ' ' +z
        else:
            sentence += ' ' +token
    return sentence.strip()

def get_token(df, col_name):
    data_token = []
    for i in df[col_name]:
        x = i.split()
        data_token += x
    return data_token

def text_distance_mra(df, col_name, string, mra_value):
    df['temp'] = np.vectorize(textdistance.mra.normalized_distance)(string, df[col_name])
    # df['temp'] = df.apply(lambda x: textdistance.mra.normalized_distance(string, x[col_name]), axis=1)
    list_str = df[df['temp']<mra_value][col_name].to_list()
    return list_str