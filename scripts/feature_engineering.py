import pandas as pd
import numpy as np
import logging


def assign_cohort(timestamp, cohorts):
    for group, dates in cohorts.items():
        start = pd.Timestamp(dates['start'])
        end = pd.Timestamp(dates['end'])
        if start <= timestamp <= end:
            return group
    return 'Other'

def bin_numerical_column(df, column, n_bins=4, labels=None):
    """
    Bin a numerical column into categories.
    """
    if labels is None:
        labels = [f'Bin {i+1}' for i in range(n_bins)]
    return pd.cut(df[column], bins=n_bins, labels=labels, include_lowest=True)


def create_features(df, cohorts):
    logging.info("Starting feature engineering.")

    # Assign cohorts
    df['Response_Group'] = df['Response_Timestamp'].apply(lambda x: assign_cohort(x, cohorts))
    df = df[df['Response_Group'] != 'Other']

    # Exclude 'Lender' from ClientUser_Type
    df = df[df['ClientUser_Type'] != 'Lender']

    # Ensure 'account_age' is calculated as the difference in days between 'Account Start Date' and 'Response_Timestamp'
    df['account_age'] = (df['Response_Timestamp'] - df['Account Start Date']).dt.days

    # Create leads_per_seat
    df['leads_per_seat'] = df['Client_#Leads'] / df['ClientUser_Cnt'].replace(0, np.nan)

    # Process Client_UserID
    df[['db_number', 'cleaned_Client_UserID']] = df['Client_UserID'].str.extract(r'(\d+)-(.+)')
    df['db_number'] = pd.to_numeric(df['db_number'], errors='coerce')

    # Create Has_Partner
    df['Has_Partner'] = df['Partner1'].fillna('No').apply(lambda x: 'No' if x == 'No' else 'Yes')

    logging.info("Feature engineering completed.")
    return df


if __name__ == "__main__":
    from data_cleaning import clean_data
    from data_loading import load_config, load_data

    config = load_config()
    ces_data = load_data(config['data']['ces_data_path'])
    ces_data_clean = clean_data(ces_data, config['preprocessing']['date_columns'],
                                config['preprocessing']['date_formats'])
    ces_data_fe = create_features(ces_data_clean, config['cohorts'])