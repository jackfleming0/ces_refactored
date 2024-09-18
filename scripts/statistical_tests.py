import pandas as pd
import logging
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def t_test_between_groups(df, group_column, target_column, group1, group2):
    group1_data = df[df[group_column] == group1][target_column].dropna()
    group2_data = df[df[group_column] == group2][target_column].dropna()
    if len(group1_data) < 2 or len(group2_data) < 2:
        logging.warning("Not enough data for t-test.")
        return None
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    logging.info(f"T-test between {group1} and {group2}: t={t_stat}, p={p_value}")
    return t_stat, p_value

def anova_test(df, group_column, target_column):
    groups = [group[target_column].dropna() for name, group in df.groupby(group_column)]
    f_statistic, p_value = stats.f_oneway(*groups)
    logging.info(f"ANOVA test: F={f_statistic}, p={p_value}")
    return f_statistic, p_value

def tukey_hsd_test(df, group_column, target_column):
    tukey = pairwise_tukeyhsd(endog=df[target_column], groups=df[group_column], alpha=0.05)
    logging.info("Performed Tukey's HSD test.")
    return tukey

if __name__ == "__main__":
    from feature_engineering import create_features
    from data_cleaning import clean_data
    from data_loading import load_config, load_data
    config = load_config()
    ces_data = load_data(config['data']['ces_data_path'])
    ces_data_clean = clean_data(ces_data, config['preprocessing']['date_columns'], config['preprocessing']['date_formats'])
    ces_data_fe = create_features(ces_data_clean, config['cohorts'])
    t_stat, p_value = t_test_between_groups(ces_data_fe, 'Response_Group', 'CES_Response_Value', 'Group 1', 'Group 2')
    f_stat, p_value = anova_test(ces_data_fe, 'Response_Group', 'CES_Response_Value')
    tukey_results = tukey_hsd_test(ces_data_fe, 'Response_Group', 'CES_Response_Value')
    print(tukey_results)