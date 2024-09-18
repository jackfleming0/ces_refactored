import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import re


def aggregate_ces_by_column(df, column_name, segment_name):
    logging.info(f"Aggregating CES by {column_name}.")
    unique_values = df[column_name].unique()
    aggregate_ces_list = []
    columns = ['CES_Score', 'Responses', 'Very Difficult', 'Difficult',
               'Somewhat Difficult', 'Neutral', 'Somewhat Easy', 'Easy',
               'Very Easy', 'Segment', 'Group']

    for value in unique_values:
        filtered_df = df[df[column_name] == value]
        ces_agg_score = filtered_df['CES_Response_Value'].mean()
        ces_agg_count = len(filtered_df)
        value_counts = filtered_df['CES_Response_Value'].value_counts()
        counts = [value_counts.get(i, 0) for i in range(1, 8)]
        temp_df = pd.DataFrame([[
            ces_agg_score, ces_agg_count, *counts, segment_name, value
        ]], columns=columns)
        aggregate_ces_list.append(temp_df)

    aggregate_ces = pd.concat(aggregate_ces_list, ignore_index=True)
    return aggregate_ces


def plot_ces_distribution(df, group_column, ces_column='CES_Response_Value', save_path=None):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x=ces_column, hue=group_column, multiple='stack', palette='viridis')
    plt.title(f'Distribution of {ces_column} by {group_column}')
    plt.xlabel(ces_column)
    plt.ylabel('Count')
    plt.legend(title=group_column)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def analyze_ces_vs_content_pages(ces_data):
    """
    This function loads CES data, uses the pre-populated CommunityPages and ContentPages columns,
    calculates correlations, and generates scatterplots to analyze the relationship between CES score
    and the content metrics. Missing data is handled by excluding rows with N/A values in those columns.
    """
    # Ensure 'CommunityPages' and 'ContentPages' columns are numeric, with N/A values converted to NaN
    ces_data['CommunityPages'] = pd.to_numeric(ces_data['CommunityPages'], errors='coerce')
    ces_data['ContentPages'] = pd.to_numeric(ces_data['ContentPages'], errors='coerce')

    # Filter out rows where either 'CommunityPages' or 'ContentPages' is NaN
    clean_data = ces_data.dropna(subset=['CommunityPages', 'ContentPages'])
    print(f"Data after dropping missing values: {clean_data.shape[0]} rows remaining.")

    # Calculate correlations between CES scores and content page metrics
    corr_community_pages = clean_data['CES_Response_Value'].corr(clean_data['CommunityPages'])
    corr_content_pages = clean_data['CES_Response_Value'].corr(clean_data['ContentPages'])

    print(f"Correlation between CES and CommunityPages: {corr_community_pages:.3f}")
    print(f"Correlation between CES and ContentPages: {corr_content_pages:.3f}")

    # Plot the relationships between CES and the content metrics
    plt.figure(figsize=(14, 6))

    # Scatter plot: CES vs CommunityPages
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=clean_data['CommunityPages'], y=clean_data['CES_Response_Value'], alpha=0.6)
    plt.title('CES vs Community Pages')
    plt.xlabel('Community Pages')
    plt.ylabel('CES Score')

    # Scatter plot: CES vs ContentPages
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=clean_data['ContentPages'], y=clean_data['CES_Response_Value'], alpha=0.6)
    plt.title('CES vs Content Pages')
    plt.xlabel('Content Pages')
    plt.ylabel('CES Score')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

    return clean_data


if __name__ == "__main__":
    from feature_engineering import create_features
    from data_cleaning import clean_data
    from data_loading import load_config, load_data

    config = load_config()
    ces_data = load_data(config['data']['ces_data_path'])
    ces_data_clean = clean_data(ces_data, config['preprocessing']['date_columns'],
                                config['preprocessing']['date_formats'])
    ces_data_fe = create_features(ces_data_clean, config['cohorts'])
    aggregate_ces = aggregate_ces_by_column(ces_data_fe, 'ClientUser_Type', 'User Type')
    plot_ces_distribution(ces_data_fe, 'ClientUser_Type')