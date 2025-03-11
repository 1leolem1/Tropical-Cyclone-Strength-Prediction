import math
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


def get_storms_speed(df):
    """with all lines of  the df, calculates a speed of movement of the storm"""
    
    for storm in df['SID'].unique():
        storm
    
    return 0


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude.
    """
    R = 6371.0  # Radius of Earth in kilometers

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c  # Distance in kilometers
    return distance


# code 

df = read_data(file_loc)
df = clean_data(df)


# print(df.head())