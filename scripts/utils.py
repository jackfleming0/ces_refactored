import pandas as pd
import logging
from dateutil.parser import parse

def parse_dates(date_str, date_formats):
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    logging.warning(f"Unable to parse date: {date_str}")
    return pd.NaT

def bin_numerical_column(df, column, n_bins=10, labels=None, binning_strategy='equal'):
    if labels is None:
        labels = [f'Bin {i+1}' for i in range(n_bins)]
    try:
        if binning_strategy == 'equal':
            df[f'{column}_binned'] = pd.cut(df[column], bins=n_bins, labels=labels, include_lowest=True)
        elif binning_strategy == 'quantile':
            df[f'{column}_binned'] = pd.qcut(df[column], q=n_bins, labels=labels, duplicates='drop')
        else:
            raise ValueError("Invalid binning_strategy. Choose 'equal' or 'quantile'.")
    except Exception as e:
        logging.error(f"Error binning column {column}: {e}")
    return df

if __name__ == "__main__":
    # Example usage
    pass