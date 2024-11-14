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


def create_db_cohort(df):
    """
    Create a new column 'db_cohort' based on Response_Timestamp and db_number.

    Categories:
    - refactored_legacy: response before 7/1/2024, db_number in [9, 11, 6, 7, 2]
    - refactored_current: response on or after 7/1/2024, db_number in [9, 11, 6, 7, 2]
    - stable_legacy: response before 7/1/2024, db_number not in [9, 11, 6, 7, 2]
    - stable_current: response on or after 7/1/2024, db_number not in [9, 11, 6, 7, 2]

    Parameters:
    df (pandas.DataFrame): Input DataFrame containing 'Response_Timestamp' and 'db_number' columns

    Returns:
    pandas.DataFrame: DataFrame with new 'db_cohort' column
    """
    # Convert Response_Timestamp to datetime if it's not already
    df['Response_Timestamp'] = pd.to_datetime(df['Response_Timestamp'])

    # Define the cutoff date
    cutoff_date = pd.Timestamp('2024-07-01')

    # Define the refactored db numbers
    refactored_dbs = [9, 11, 6, 7, 2]

    # Create the db_cohort column
    conditions = [
        (df['Response_Timestamp'] < cutoff_date) & (df['db_number'].isin(refactored_dbs)),
        (df['Response_Timestamp'] >= cutoff_date) & (df['db_number'].isin(refactored_dbs)),
        (df['Response_Timestamp'] < cutoff_date) & (~df['db_number'].isin(refactored_dbs)),
        (df['Response_Timestamp'] >= cutoff_date) & (~df['db_number'].isin(refactored_dbs))
    ]
    choices = ['refactored_legacy', 'refactored_current', 'stable_legacy', 'stable_current']

    df['db_cohort'] = np.select(conditions, choices, default='unknown')

    return df


def add_ad_spend_features(df, ad_spend_path, config):
    """
    Add ad spend features to CES data with both quintile and predefined binning.

    Parameters:
    df (pd.DataFrame): CES data with ClientSite_Name and Response_Timestamp
    ad_spend_path (str): Path to ad spend CSV file
    config (dict): Configuration dictionary containing binning parameters

    Returns:
    pd.DataFrame: Original dataframe with new ad spend columns and binning
    """
    # Load ad spend data
    ad_spend = pd.read_csv(ad_spend_path)

    # Convert CES timestamp to same format as ad spend data
    df['MonthYear'] = pd.to_datetime(df['Response_Timestamp']).dt.strftime('%b-%y')

    # Create matching key in both dataframes
    df['matching_key'] = df['ClientSite_Name'] + '_' + df['MonthYear']
    ad_spend['matching_key'] = ad_spend['Account name'] + '_' + ad_spend['Month']

    # Create list of matches
    matching_keys = set(ad_spend['matching_key'])
    df['AdSpendYN'] = df['matching_key'].isin(matching_keys)

    # Print match statistics
    print(f"\nTotal CES records: {len(df)}")
    print(f"Records with matching ad spend: {df['AdSpendYN'].sum()}")

    # Select and rename columns from ad spend data
    ad_spend_cols = {
        'Cost': 'AdSpendCost',
        'Clicks': 'AdSpendClicks',
        'Impr.': 'AdSpendImpressions',
        'CTR': 'AdSpendCTR',
        'Avg. CPC': 'AdSpendCPC',
        'Conversions': 'AdSpendConversions',
        'Cost / conv.': 'AdSpendCostPerConversion',
        'Conv. rate': 'AdSpendConversionRate',
        'Search impr. share': 'AdSpendSearchImprShare',
        'Search lost IS (budget)': 'AdSpendSearchLost'
    }

    # Create subset of ad spend data with renamed columns
    ad_spend_subset = ad_spend[['matching_key'] + list(ad_spend_cols.keys())].copy()
    ad_spend_subset = ad_spend_subset.rename(columns=ad_spend_cols)

    # Function to safely convert percentages
    def clean_percentage(value):
        if pd.isna(value):
            return value
        if isinstance(value, str):
            # Handle '< 10%' case
            if '<' in value:
                return float(value.replace('< ', '').rstrip('%')) / 100
            return float(value.rstrip('%')) / 100
        return value

    # Function to safely convert currency
    def clean_currency(value):
        if pd.isna(value):
            return value
        if isinstance(value, str):
            return float(value.replace('$', '').replace(',', ''))
        return value

    # Clean up percentage columns
    percentage_cols = ['AdSpendCTR', 'AdSpendConversionRate', 'AdSpendSearchImprShare', 'AdSpendSearchLost']
    for col in percentage_cols:
        if col in ad_spend_subset.columns:
            ad_spend_subset[col] = ad_spend_subset[col].apply(clean_percentage)

    # Clean up currency columns
    currency_cols = ['AdSpendCost', 'AdSpendCPC', 'AdSpendCostPerConversion']
    for col in currency_cols:
        if col in ad_spend_subset.columns:
            ad_spend_subset[col] = ad_spend_subset[col].apply(clean_currency)

    # Merge while keeping all CES records
    df = df.merge(ad_spend_subset, on='matching_key', how='left')

    # Add binning for AdSpend Cost (only for records with ad spend)
    spend_mask = df['AdSpendYN'] & df['AdSpendCost'].notna()
    if spend_mask.any():
        # Quintile binning
        df.loc[spend_mask, 'AdSpend_Quintile'] = pd.qcut(
            df.loc[spend_mask, 'AdSpendCost'],
            q=config['ad_spend_analysis']['quantile_bins'],
            labels=False,
            duplicates='drop'
        ).add(1)  # Add 1 to make quintiles 1-based

        # Predefined binning - using upper bounds
        bins = config['ad_spend_analysis']['spend_brackets']['bins']
        df.loc[spend_mask, 'AdSpend_Bins'] = pd.cut(
            df.loc[spend_mask, 'AdSpendCost'],
            bins=bins,
            labels=[b for b in bins[1:]],  # Use upper bounds as labels
            include_lowest=True
        )

    # Add binning for Cost Per Lead (only for records with ad spend and conversions)
    cpl_mask = df['AdSpendYN'] & df['AdSpendCostPerConversion'].notna()
    if cpl_mask.any():
        # Quintile binning
        df.loc[cpl_mask, 'AdSpendCPL_Quintile'] = pd.qcut(
            df.loc[cpl_mask, 'AdSpendCostPerConversion'],
            q=config['ad_spend_analysis']['quantile_bins'],
            labels=False,
            duplicates='drop'
        ).add(1)  # Add 1 to make quintiles 1-based

        # Predefined binning - using upper bounds
        cpl_bins = config['ad_spend_analysis']['cost_per_lead_brackets']['bins']
        df.loc[cpl_mask, 'AdSpendCPL_Bins'] = pd.cut(
            df.loc[cpl_mask, 'AdSpendCostPerConversion'],
            bins=cpl_bins,
            labels=[b for b in cpl_bins[1:]],  # Use upper bounds as labels
            include_lowest=True
        )

    # Clean up
    df = df.drop(['matching_key', 'MonthYear'], axis=1)

    return df

def create_features(df, cohorts, config=None, ad_spend_path=None):
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

    create_db_cohort(df)

    # Add ad spend features if config and path are provided
    if config is not None and ad_spend_path is not None:
        logging.info("Adding ad spend features.")
        df = add_ad_spend_features(df, ad_spend_path, config)
    else:
        logging.info("Skipping ad spend features due to missing config or path.")

    logging.info("Feature engineering completed.")
    return df

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