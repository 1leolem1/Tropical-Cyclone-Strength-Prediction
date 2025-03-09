import pandas as pd


file_loc = 'data/ibtracs.csv'

def read_data(file_loc):
    data = pd.read_csv(file_loc)
    return data

print(read_data(file_loc))