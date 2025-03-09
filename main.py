import pandas as pd

file_loc = 'data/ibtracs.csv'

def read_data(file_loc, low_memory=False):
    dataframe = pd.read_csv(file_loc)
    dataframe = dataframe.drop(index=0)
    dataframe['ISO_TIME'] = pd.to_datetime(dataframe['ISO_TIME'])
    return dataframe

def clean_data(df):
    df = df[df['ISO_TIME'] < '1990-01-01'] # Filter out data after 1989 (where we have a target variable)
    return df


# code 

df = read_data(file_loc)
df = clean_data(df)
