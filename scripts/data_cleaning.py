import pandas as pd
import numpy as np
import logging
from utils import parse_dates


def clean_data(df, date_columns, date_formats):
    logging.info("Starting data cleaning process.")

    # Drop rows with missing CES_Response_Value
    df = df.dropna(subset=['CES_Response_Value'])
    logging.info(f"After dropping missing CES_Response_Value: {df.shape}")

    # Parse date columns
    for col in date_columns:
        df[col] = df[col].apply(lambda x: parse_dates(x, date_formats))
        missing_dates = df[col].isna().sum()
        logging.info(f"Column '{col}' has {missing_dates} missing dates after parsing.")

    # Drop rows with missing dates
    df = df.dropna(subset=date_columns)
    logging.info(f"After dropping rows with missing dates: {df.shape}")

    return df


if __name__ == "__main__":
    from data_loading import load_config, load_data

    config = load_config()
    ces_data = load_data(config['data']['ces_data_path'])
    ces_data_clean = clean_data(ces_data, config['preprocessing']['date_columns'],
                                config['preprocessing']['date_formats'])