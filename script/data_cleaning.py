import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

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
    for suffix in ['_PRES', '_CAT', '_GRADE', '_CI']: 
        new_column_name = f'NEW_{suffix.lstrip("_")}'
        merged_column, original_columns = merge_columns_with_suffix(df, suffix, columns_to_merge)

        if merged_column is not None: 
            df[new_column_name] = merged_column
            merged_columns_info[new_column_name] = original_columns

    # Drop the original columns that were merged
    columns_to_drop = [col for cols in merged_columns_info.values() for col in cols]
    df = df.drop(columns=columns_to_drop)

    print("\nMerged columns info:")
    for new_col, original_cols in merged_columns_info.items():
        print(f"{new_col} was created by merging: {original_cols}")

    return df

def save_df(df):
    # Saving the dataframe
    df.to_csv('./data/ibtracs_clean.csv', index=False)

    # Print the final DataFrame
    print("\nFinal DataFrame:")
    print(df)

def preprocess_temporal_variables(df, target_variable='TD9636_STAGE'):
    # Create start date and age of the storm
    df['start_date'] = df.groupby('SID')['ISO_TIME'].transform('min')
    df['age_hours'] = (df['ISO_TIME'] - df['start_date']).dt.total_seconds() / 3600

    # Extract YEAR, MONTH, and start_month from ISO_TIME
    df['YEAR'] = df['ISO_TIME'].dt.year
    df['MONTH'] = df['ISO_TIME'].dt.month
    df['start_month'] = df['start_date'].dt.month

    # Define temporal variables
    variables_temporelles = ["start_date", "age_hours", "SEASON", "ISO_TIME", "YEAR", "MONTH", "start_month"]

    # Convert target variable to numeric
    df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce')

    # Drop rows with missing target variable
    df = df.dropna(subset=[target_variable])

    return df, variables_temporelles

def determine_averaging_period(agency):
    mapping = {
        'hurdat_atl': '1min',
        'hurdat_epa': '1min',
        'cphc': '1min',
        'tokyo': '10min',
        'newdelhi': '3min',
        'reunion': '10min',
        'bom': '10min',
        'nadi': '10min',
        'wellington': '10min'
    }
    return mapping.get(agency, '10min')  # Default to 10 min if not specified

def apply_wind_conversion(row):
    # Convert to numeric to handle string values
    dist = pd.to_numeric(row['DIST2LAND'], errors='coerce') 
    wind_speed = pd.to_numeric(row['WMO_WIND'], errors='coerce') 
    agency = row['WMO_AGENCY']

    # Check if dist or wind_speed is NaN
    if pd.isna(dist) or pd.isna(wind_speed):
        return np.nan

    # Determine the averaging period
    avg_period = determine_averaging_period(agency)

    # Determine the exposure category based on DIST2LAND
    if dist > 20:
        exposure_ratios = {'1min': 1, '3min': 1, '10min': 1.05}
    elif dist <= 1:
        exposure_ratios = {'1min': 1, '3min': 1.10, '10min': 1.16}
    else:
        exposure_ratios = {'1min': 1, '3min': 1.05, '10min': 1.11}

    return wind_speed * exposure_ratios[avg_period]

def adjust_wind_speed(df):
    # Ensure relevant columns are numeric before applying the function
    df['DIST2LAND'] = pd.to_numeric(df['DIST2LAND'], errors='coerce')
    df['WMO_WIND'] = pd.to_numeric(df['WMO_WIND'], errors='coerce')

    df['WMO_WIND_ADJUSTED'] = df.apply(apply_wind_conversion, axis=1)

    return df

def fill_missing_wind_data(df):
    # Replace missing values with TD9636_WIND
    df['WMO_WIND_ADJUSTED_COMPLETED'] = df['WMO_WIND_ADJUSTED'].fillna(df['TD9636_WIND'])

    # Convert to numeric
    df['WMO_WIND_ADJUSTED_COMPLETED'] = pd.to_numeric(df['WMO_WIND_ADJUSTED_COMPLETED'], errors='coerce')

    # Sort by SID and age_hours
    df = df.sort_values(by=['SID', 'age_hours'])

    # Forward and backward fill
    df['forward'] = df.groupby('SID')['WMO_WIND_ADJUSTED_COMPLETED'].ffill()
    df['backward'] = df.groupby('SID')['WMO_WIND_ADJUSTED_COMPLETED'].bfill()

    def fill_value(row):
        original = row['WMO_WIND_ADJUSTED_COMPLETED']
        if not pd.isna(original):
            return original
        
        f = row['forward']
        b = row['backward']
        
        if pd.isna(f) and pd.isna(b):
            return np.nan    
        if pd.isna(b):
            return f    
        if pd.isna(f):
            return b    
        return (f + b) / 2

    df['WMO_WIND_ADJUSTED_COMPLETED'] = df.apply(fill_value, axis=1)
    df.drop(columns=['forward', 'backward'], inplace=True)

    return df

def drop_unnecessary_columns(df):
    # Drop unnecessary wind columns
    df = df.drop(columns=["WMO_WIND_ADJUSTED", "WMO_WIND", "TD9636_WIND", "USA_WIND", "TOKYO_WIND", "CMA_WIND", "HKO_WIND", "NEUMANN_WIND"])

    # Drop columns with more than 50% missing values
    threshold = 0.5 
    nan_ratio = df.isna().mean()
    df = df.loc[:, nan_ratio < threshold]

    # Drop other unnecessary columns
    df = df.drop(columns=["SID", "NUMBER", "SUBBASIN", "NAME", "USA_ATCF_ID"])

    return df

def final_data_cleaning(df):
    # Convert STORM_SPEED to numeric
    df['STORM_SPEED'] = pd.to_numeric(df['STORM_SPEED'], errors='coerce').astype('Int64')

    # Convert categorical columns
    for col in ['BASIN', 'NATURE', 'TRACK_TYPE', 'LANDFALL', 'IFLAG', 'STORM_DIR']:
        df[col] = df[col].astype('category')

    return df

#------------MAIN EXCUTION------------
if __name__=="__main__":
    file_path = "./data/ibtracs.csv" 
    
    # Create a dataframe with correct types
    df = string_to_date(file_path)

    # The target column "TD9636_STAGE" only has data from 1980 to 1989, so we extract that period to a dataframe
    df = extract_pertinent_data(df)

    # Convert missing values from ' ' to NA
    df = convert_missing_values(df)
    
    # Preprocess temporal variables
    df, variables_temporelles = preprocess_temporal_variables(df)

    # Adjust wind speed
    df = adjust_wind_speed(df)

    # Fill missing wind data
    df = fill_missing_wind_data(df)

    # Drop unnecessary columns
    df = drop_unnecessary_columns(df)

    # Remove columns with 50% missing data
    df = remove_na_columns(df, threshold=0.5)
    
    # Final data cleaning
    df = final_data_cleaning(df)

    # Save the cleaned dataframe
    save_df(df)