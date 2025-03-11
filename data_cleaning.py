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



def create_storm_speed_feature(df):

    """
    DEPENDS ON haversine() function
    """


    df['LAT'] = df['LAT'].astype(float)
    df['LON'] = df['LON'].astype(float)


    df = df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)
    df['LAT_PREV']       = df['LAT'].shift(1)
    df['LON_PREV']       = df['LON'].shift(1)
    df['ISO_TIME_PREV']  = df['ISO_TIME'].shift(1)
    df['is_first_measure'] = df['SID'].ne(df['SID'].shift())
    df['STORM_SPEED'] = 0.0

    for idx, row in df.iterrows():
        if row['is_first_measure']:
            continue
        else:
            dist = haversine(row['LAT'], row['LON'],
                             row['LAT_PREV'], row['LON_PREV'])
            time_delta = (row['ISO_TIME'] - row['ISO_TIME_PREV']).total_seconds() / 3600.0
            speed = dist / time_delta if time_delta != 0 else 0.0
            df.at[idx, 'STORM_SPEED'] = speed
    return df


# code 

df = read_data(file_loc)
df = clean_data(df)
df  = create_storm_speed_feature(df)
print(df['STORM_SPEED'].head())
# print(df.head())