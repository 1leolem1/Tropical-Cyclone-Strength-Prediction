import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

#------------FUNCTIONS------------
def string_to_date(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df = df.drop(index=0)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    return df

def extract_pertinent_data(df):
    df = df[df['ISO_TIME'] < '1990-01-01']
    return df

def remove_na_columns(df, threshold="all"):
    df.replace(to_replace=" ", value=pd.NA, inplace=True)

    if threshold == "all":
        df = df.dropna(axis=1, how='all')
    elif isinstance(threshold, float) and 0 <= threshold <= 1:
        df = df.dropna(axis=1, thresh=len(df) * (1 - threshold))
    else:
        raise ValueError("Threshold must be 'all' or a float between 0 and 1.")

    return df

def convert_missing_values(df):
    df.replace(to_replace=" ", value=pd.NA, inplace=True)
    return df

def visualize_data(df):
    plt.figure(figsize=(10,6))
    msno.bar(df, sort="descending")
    plt.show()

def columns_with_more_than_50_percent_missing(df):
    missing_percentage = df.isna().mean()
    columns_with_more_than_50_percent_missing = missing_percentage[missing_percentage > 0.5].index.tolist()
    return columns_with_more_than_50_percent_missing

def merge_columns_with_suffix(df, suffix, columns_to_merge):
    columns_for_suffix = [col for col in columns_to_merge if col.endswith(suffix)]

    if columns_for_suffix: 
        merged_column = df[columns_for_suffix].bfill(axis=1).iloc[:, 0]
        return merged_column, columns_for_suffix
    else:
        return None, []

def merge_columns(df):
    # Identify columns with more than 50% missing values
    columns_to_merge = columns_with_more_than_50_percent_missing(df)
    print("Columns with more than 50% missing values:", columns_to_merge)

    # Merge columns with the same suffix
    merged_columns_info = {} 
    for suffix in ['_LAT', '_LON', '_WIND', '_PRES', '_CAT', '_GRADE', '_CI']: 
        new_column_name = f'NEW_{suffix.lstrip("_")}'
        merged_column, original_columns = merge_columns_with_suffix(df, suffix, columns_to_merge)

        if merged_column is not None: 
            df[new_column_name] = merged_column
            merged_columns_info[new_column_name] = original_columns

    # Drop the original columns that were merged
    columns_to_drop = [col for cols in merged_columns_info.values() for col in cols]
    df = df.drop(columns=columns_to_drop)

    # Saving the dataframe
    df.to_csv('./Tropical-Cyclone-Strength-Prediction-main/data/ibtracs_clean.csv', index=False)

    # Print the final DataFrame and merged columns info
    print("\nFinal DataFrame:")
    print(df)

    print("\nMerged columns info:")
    for new_col, original_cols in merged_columns_info.items():
        print(f"{new_col} was created by merging: {original_cols}")

#------------MAIN EXCUTION------------
if __name__=="__main__":
    file_path = "./Tropical-Cyclone-Strength-Prediction-main/data/ibtracs.csv" 
    
    # Create a dataframe with correct types
    df = string_to_date(file_path)

    # The target column "TD9636_STAGE" only has data from 1980 to 1989, so we extract that period to a dataframe
    df = extract_pertinent_data(df)

    # Remove columns with 100% missing data
    df = remove_na_columns(df, threshold="all")

    # Convert missing values from ' ' to NA
    df = convert_missing_values(df)
    
    # Merge columns with more than 50% of missing values into a "NEW" column by category (assuming they are the same)
    df = merge_columns(df)