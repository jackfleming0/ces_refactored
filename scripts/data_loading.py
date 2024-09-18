import pandas as pd
import logging
import yaml
import os

def load_config(config_path='../config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(file_path):
    logging.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(df, columns_to_bin=None):
    # Copy 'ka' into 'FE_Site_ID' where 'FE_Site_ID' is missing or blank
    df['FE_Site_ID'] = df['FE_Site_ID'].fillna(df['ka'])  # Fill missing FE_Site_ID with 'ka' values
    df['FE_Site_ID'] = df['FE_Site_ID'].replace('', df['ka'])  # Replace empty strings with 'ka' values

    # Assign cohorts
    df['Response_Group'] = df['Response_Timestamp'].apply(assign_cohort)
    df = df[df['Response_Group'] != '']  # Filter out empty Response_Group
    df = df[df['ClientUser_Type'] != 'Lender']  # Filter out Lender

    # Cohorts are ordered based on their timestamp
    cohorts = get_cohorts()
    cohort_order = sorted(cohorts.keys(), key=lambda x: pd.Timestamp(cohorts[x][0]))
    df['Response_Group'] = pd.Categorical(df['Response_Group'], categories=cohort_order, ordered=True)

    # If binning is needed, bin the specified columns
    if columns_to_bin:
        for column, n_bins in columns_to_bin.items():
            df[f'{column}_binned'] = bin_numerical_column(df, column, n_bins)

    return df

if __name__ == "__main__":
    config = load_config()
    ces_data = load_data(config['data']['ces_data_path'])