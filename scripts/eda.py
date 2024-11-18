import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
import re
from statistical_tests import t_test_between_groups
from scipy import stats
from utils import OutputFormatter

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

# Helper function to clean numeric values
def clean_numeric_value(value):
    """Convert various numeric formats to float."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Remove currency and comma formatting
        cleaned = value.replace('$', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    return np.nan


def analyze_ces_vs_content_pages(ces_data):
    """
    This function loads CES data, uses the pre-populated CommunityPages and ContentPages columns,
    calculates correlations, and generates scatterplots to analyze the relationship between CES score
    and the content metrics. Missing data is handled by excluding rows with N/A values in those columns.
    """
    # Print data types before conversion
    print("Data types before conversion:")
    print(ces_data[['CommunityPages', 'ContentPages', 'CES_Response_Value']].dtypes)

    # Print some sample values to see what we're working with
    print("\nSample values before conversion:")
    print(ces_data[['CommunityPages', 'ContentPages']].head(10))

    # Ensure 'CommunityPages' and 'ContentPages' columns are numeric, with N/A values converted to NaN
    ces_data['CommunityPages'] = pd.to_numeric(ces_data['CommunityPages'], errors='coerce')
    ces_data['ContentPages'] = pd.to_numeric(ces_data['ContentPages'], errors='coerce')
    ces_data['CES_Response_Value'] = pd.to_numeric(ces_data['CES_Response_Value'], errors='coerce')

    # Print data types after conversion
    print("\nData types after conversion:")
    print(ces_data[['CommunityPages', 'ContentPages', 'CES_Response_Value']].dtypes)

    # Filter out rows where either 'CommunityPages' or 'ContentPages' is NaN
    clean_data = ces_data.dropna(subset=['CommunityPages', 'ContentPages', 'CES_Response_Value'])
    print(f"\nData after dropping missing values: {clean_data.shape[0]} rows remaining.")

    # Verify we have numeric data before correlation
    print("\nValue counts after cleaning:")
    print("CommunityPages unique values:", clean_data['CommunityPages'].unique())
    print("ContentPages unique values:", clean_data['ContentPages'].unique())
    print("CES_Response_Value unique values:", clean_data['CES_Response_Value'].unique())

    # Calculate correlations between CES scores and content page metrics
    corr_community_pages = clean_data['CES_Response_Value'].corr(clean_data['CommunityPages'])
    corr_content_pages = clean_data['CES_Response_Value'].corr(clean_data['ContentPages'])

    print(f"\nCorrelation between CES and CommunityPages: {corr_community_pages:.3f}")
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


def analyze_cohort_comparisons(df, config, group_column='Response_Group', value_column='CES_Response_Value'):
    """
    Performs pairwise comparisons between all cohort groups and formats the results.

    Args:
        df: DataFrame containing the data
        config: Configuration dictionary containing cohort definitions
        group_column: Column containing cohort group labels
        value_column: Column containing the values to compare
    """
    # Get cohorts from config and sort them by start date
    cohort_info = []
    for group, dates in config['cohorts'].items():
        cohort_info.append({
            'group': group,
            'start_date': pd.to_datetime(dates['start'])
        })

    # Sort cohorts by start date
    cohort_info = sorted(cohort_info, key=lambda x: x['start_date'])
    cohorts = [c['group'] for c in cohort_info]

    # Verify all cohorts exist in the data
    existing_cohorts = df[group_column].unique()
    cohorts = [c for c in cohorts if c in existing_cohorts]

    results = []
    # Perform all pairwise comparisons
    for i, cohort1 in enumerate(cohorts):
        for cohort2 in cohorts[i + 1:]:  # Compare with all subsequent cohorts
            t_stat, p_value = t_test_between_groups(
                df,
                group_column,
                value_column,
                cohort1,
                cohort2
            )

            # Calculate mean values and sample sizes for each group
            group1_stats = df[df[group_column] == cohort1][value_column]
            group2_stats = df[df[group_column] == cohort2][value_column]

            results.append({
                'comparison': f"{cohort1} vs {cohort2}",
                'cohort1': cohort1,
                'cohort2': cohort2,
                'cohort1_n': len(group1_stats),
                'cohort2_n': len(group2_stats),
                'cohort1_mean': group1_stats.mean(),
                'cohort2_mean': group2_stats.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohort1_start': config['cohorts'][cohort1]['start'],
                'cohort2_start': config['cohorts'][cohort2]['start']
            })

    # Format and print results
    print(OutputFormatter.format_cohort_comparison_results(results))

    return results


def analyze_ces_vs_ad_spend(ces_data):
    """
    Comprehensive analysis of relationships between CES and ad spend metrics.

    Returns:
    tuple: (ad_spend_data, correlations, stats_summary)
    - ad_spend_data: DataFrame with only records that have ad spend
    - correlations: Dictionary of correlations between CES and ad spend metrics
    - stats_summary: Dictionary of key statistical findings
    """
    print("\n=== Ad Spend Analysis ===")

    # Only analyze records with ad spend
    ad_spend_data = ces_data[ces_data['AdSpendYN']].copy()
    print(f"Analyzing {len(ad_spend_data)} records with ad spend data")

    # Store all our findings
    correlations = {}
    stats_summary = {}

    # Ensure CES_Response_Value is numeric
    ad_spend_data['CES_Response_Value'] = pd.to_numeric(ad_spend_data['CES_Response_Value'], errors='coerce')

    # 1. Basic correlation analysis
    spend_metrics = [
        'AdSpendCost', 'AdSpendConversions', 'AdSpendCTR',
        'AdSpendCPC', 'AdSpendCostPerConversion', 'AdSpendSearchImprShare'
    ]

    for metric in spend_metrics:
        if metric in ad_spend_data.columns:
            # Convert to numeric if string, handling any special characters
            if pd.api.types.is_string_dtype(ad_spend_data[metric]):
                value = ad_spend_data[metric].str.replace('%', '', regex=False)
                value = value.str.replace('$', '', regex=False)
                value = value.str.replace(',', '', regex=False)
                value = value.str.replace('<', '', regex=False)
                ad_spend_data[metric] = pd.to_numeric(value, errors='coerce')

            if pd.api.types.is_numeric_dtype(ad_spend_data[metric]):
                corr = ad_spend_data['CES_Response_Value'].corr(ad_spend_data[metric])
                correlations[metric] = corr
                print(f"\nCorrelation between CES and {metric}: {corr:.3f}")

    # 2. Per-seat analysis
    if 'ClientUser_Cnt' in ad_spend_data.columns:
        ad_spend_data['ClientUser_Cnt'] = pd.to_numeric(ad_spend_data['ClientUser_Cnt'], errors='coerce')
        valid_seats = ad_spend_data['ClientUser_Cnt'] > 0

        if valid_seats.any():
            ad_spend_data.loc[valid_seats, 'ConversionsPerSeat'] = (
                    ad_spend_data.loc[valid_seats, 'AdSpendConversions'] /
                    ad_spend_data.loc[valid_seats, 'ClientUser_Cnt']
            )
            ad_spend_data.loc[valid_seats, 'SpendPerSeat'] = (
                    ad_spend_data.loc[valid_seats, 'AdSpendCost'] /
                    ad_spend_data.loc[valid_seats, 'ClientUser_Cnt']
            )

            correlations['ConversionsPerSeat'] = ad_spend_data['CES_Response_Value'].corr(
                ad_spend_data['ConversionsPerSeat']
            )
            correlations['SpendPerSeat'] = ad_spend_data['CES_Response_Value'].corr(
                ad_spend_data['SpendPerSeat']
            )

    # 3. Quintile Analysis
    if 'AdSpendCost' in ad_spend_data.columns:
        valid_data = ad_spend_data.dropna(subset=['AdSpendCost', 'CES_Response_Value'])
        if len(valid_data) > 0:
            quintiles = pd.qcut(valid_data['AdSpendCost'], q=5, labels=False) + 1
            quintile_groups = {str(q): group['CES_Response_Value'].values
                               for q, group in valid_data.groupby(quintiles)}

            # Store quintile means and group data
            stats_summary['quintile_means'] = valid_data.groupby(quintiles)['CES_Response_Value'].mean().to_dict()
            stats_summary['quintile_groups'] = quintile_groups

            if len(quintile_groups) > 1:
                f_stat, p_value = stats.f_oneway(*quintile_groups.values())
                stats_summary['quintile_anova'] = {'f_stat': f_stat, 'p_value': p_value}

    # 4. Bucket Analysis
    if 'AdSpend_Bins' in ad_spend_data.columns:
        valid_data = ad_spend_data.dropna(subset=['AdSpend_Bins', 'CES_Response_Value'])
        if len(valid_data) > 0:
            # Calculate means per bucket
            bucket_means = valid_data.groupby('AdSpend_Bins')['CES_Response_Value'].mean()
            stats_summary['bucket_means'] = bucket_means.to_dict()

            # Calculate sizes per bucket
            bucket_sizes = valid_data.groupby('AdSpend_Bins')['CES_Response_Value'].count()
            stats_summary['bucket_sizes'] = bucket_sizes.to_dict()

            # Store the groups for ANOVA
            bucket_groups = [group['CES_Response_Value'].values
                             for _, group in valid_data.groupby('AdSpend_Bins')]
            if len(bucket_groups) > 1:
                f_stat, p_value = stats.f_oneway(*bucket_groups)
                stats_summary['bucket_anova'] = {'f_stat': f_stat, 'p_value': p_value}

    # 5. Full Population Efficiency Categorization
    if 'AdSpendCost' in ad_spend_data.columns and 'AdSpendConversions' in ad_spend_data.columns:
        # Load full ad spend dataset
        full_ad_spend = pd.read_csv('../data/All Accounts-Table 1.csv')

        # Clean the full dataset columns
        full_ad_spend['Cost'] = full_ad_spend['Cost'].apply(clean_numeric_value)
        full_ad_spend['Conversions'] = pd.to_numeric(full_ad_spend['Conversions'], errors='coerce')

        # Calculate medians from full population
        full_spend_median = full_ad_spend['Cost'].median()
        full_conv_median = full_ad_spend['Conversions'].median()

        print("\nDebug - Creating efficiency categories...")
        print(f"Spend median: ${full_spend_median:,.2f}")
        print(f"Conversion median: {full_conv_median:.1f}")

        # Create categories for CES respondents
        conditions = [
            (ad_spend_data['AdSpendCost'] > full_spend_median) & (
                        ad_spend_data['AdSpendConversions'] > full_conv_median),
            (ad_spend_data['AdSpendCost'] > full_spend_median) & (
                        ad_spend_data['AdSpendConversions'] <= full_conv_median),
            (ad_spend_data['AdSpendCost'] <= full_spend_median) & (
                        ad_spend_data['AdSpendConversions'] > full_conv_median),
            (ad_spend_data['AdSpendCost'] <= full_spend_median) & (
                        ad_spend_data['AdSpendConversions'] <= full_conv_median)
        ]
        choices = ['High Spend, High Conv.', 'High Spend, Low Conv.',
                   'Low Spend, High Conv.', 'Low Spend, Low Conv.']

        ad_spend_data['Efficiency_Category'] = np.select(conditions, choices, default='Uncategorized')

        print("\nDebug - Category Distribution:")
        print(ad_spend_data['Efficiency_Category'].value_counts())

        # Calculate stats for each category
        eff_stats = {}
        for category in choices:
            category_data = ad_spend_data[ad_spend_data['Efficiency_Category'] == category]
            if len(category_data) > 0:
                eff_stats[category] = {
                    'mean': category_data['CES_Response_Value'].mean(),
                    'n': len(category_data)
                }

        stats_summary['efficiency_stats'] = eff_stats

        # Calculate population distribution
        full_conditions = [
            (full_ad_spend['Cost'] > full_spend_median) & (full_ad_spend['Conversions'] > full_conv_median),
            (full_ad_spend['Cost'] > full_spend_median) & (full_ad_spend['Conversions'] <= full_conv_median),
            (full_ad_spend['Cost'] <= full_spend_median) & (full_ad_spend['Conversions'] > full_conv_median),
            (full_ad_spend['Cost'] <= full_spend_median) & (full_ad_spend['Conversions'] <= full_conv_median)
        ]

        full_ad_spend['Efficiency_Category'] = np.select(full_conditions, choices, default='Uncategorized')
        pop_distribution = full_ad_spend['Efficiency_Category'].value_counts()
        stats_summary['population_distribution'] = pop_distribution.to_dict()

        print("\nDebug - Population Distribution:")
        print(pop_distribution)

        # Calculate ANOVA if enough data
        groups = [group['CES_Response_Value'].values
                  for _, group in ad_spend_data.groupby('Efficiency_Category')
                  if len(group) > 0]

        if len(groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                stats_summary['efficiency_anova'] = {'f_stat': f_stat, 'p_value': p_value}
            except Exception as e:
                print(f"\nDebug - ANOVA error: {e}")
                stats_summary['efficiency_anova'] = {'f_stat': np.nan, 'p_value': np.nan}

        print("\nDebug - Stats Summary Keys:", stats_summary.keys())
        if 'efficiency_stats' in stats_summary:
            print("\nDebug - Efficiency Stats:", stats_summary['efficiency_stats'])

    # 6. CES Respondent-Only Efficiency Categorization
    if 'AdSpendCost' in ad_spend_data.columns and 'AdSpendConversions' in ad_spend_data.columns:
        # Get only records with CES responses
        ces_respondents = ad_spend_data.dropna(subset=['CES_Response_Value'])

        # Calculate medians from CES respondents only
        ces_spend_median = ces_respondents['AdSpendCost'].median()
        ces_conv_median = ces_respondents['AdSpendConversions'].median()

        print("\nDebug - CES Respondent Medians:")
        print(f"Spend median: ${ces_spend_median:,.2f}")
        print(f"Conversion median: {ces_conv_median:.1f}")

        # Create categories using CES respondent thresholds
        ces_conditions = [
            (ces_respondents['AdSpendCost'] > ces_spend_median) & (
                        ces_respondents['AdSpendConversions'] > ces_conv_median),
            (ces_respondents['AdSpendCost'] > ces_spend_median) & (
                        ces_respondents['AdSpendConversions'] <= ces_conv_median),
            (ces_respondents['AdSpendCost'] <= ces_spend_median) & (
                        ces_respondents['AdSpendConversions'] > ces_conv_median),
            (ces_respondents['AdSpendCost'] <= ces_spend_median) & (
                        ces_respondents['AdSpendConversions'] <= ces_conv_median)
        ]

        ces_respondents['CES_Only_Efficiency'] = np.select(ces_conditions, choices, default='Uncategorized')

        # Calculate stats
        ces_eff_stats = {}
        ces_eff_groups = []

        for category in choices:
            category_data = ces_respondents[ces_respondents['CES_Only_Efficiency'] == category]
            if len(category_data) > 0:
                ces_group = category_data['CES_Response_Value'].dropna()
                if len(ces_group) > 0:
                    ces_eff_stats[category] = {
                        'mean': ces_group.mean(),
                        'n': len(ces_group)
                    }
                    ces_eff_groups.append(ces_group.values)

        # Store results
        stats_summary['ces_only_efficiency_stats'] = ces_eff_stats
        stats_summary['ces_thresholds'] = {
            'spend_threshold': ces_spend_median,
            'conv_threshold': ces_conv_median
        }

        # Calculate ANOVA if possible
        if len(ces_eff_groups) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*ces_eff_groups)
                stats_summary['ces_only_efficiency_anova'] = {'f_stat': f_stat, 'p_value': p_value}
            except Exception as e:
                print(f"\nDebug - CES-only ANOVA error: {e}")
                stats_summary['ces_only_efficiency_anova'] = {'f_stat': np.nan, 'p_value': np.nan}

    return ad_spend_data, correlations, stats_summary


def analyze_ad_spend_difference(df):
    """Compare CES scores between respondents with and without ad spend."""
    # Ensure CES_Response_Value is numeric
    df['CES_Response_Value'] = pd.to_numeric(df['CES_Response_Value'], errors='coerce')

    # Split into groups
    with_ad_spend = df[df['AdSpendYN']]['CES_Response_Value'].dropna()
    without_ad_spend = df[~df['AdSpendYN']]['CES_Response_Value'].dropna()

    # Calculate basic statistics
    stats_dict = {
        'With Ad Spend': {
            'n': len(with_ad_spend),
            'mean': with_ad_spend.mean(),
            'std': with_ad_spend.std()
        },
        'Without Ad Spend': {
            'n': len(without_ad_spend),
            'mean': without_ad_spend.mean(),
            'std': without_ad_spend.std()
        }
    }

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(with_ad_spend, without_ad_spend)

    # Print results
    print("\nAd Spend vs No Ad Spend Analysis:")
    print("-" * 50)
    print(f"{'Group':20} {'n':>6} {'Mean CES':>10} {'Std Dev':>10}")
    print("-" * 50)
    for group, data in stats_dict.items():
        print(f"{group:20} {data['n']:6d} {data['mean']:10.2f} {data['std']:10.2f}")

    print(f"\nT-test results: t={t_stat:.2f}, p={p_value:.4f}")

    return t_stat, p_value, stats_dict

def format_ad_analysis_results(correlations, stats_summary):
    """Format analysis results for clear display."""
    print("\n=== Ad Spend Analysis Results ===")
    print(f"Total records analyzed: {sum(stats_summary['bucket_sizes'].values())}")

    # Correlations section
    print("\nCorrelations with CES (sorted by strength):")
    print(f"{'Metric':30} {'Correlation':>10}")
    print("-" * 42)
    for metric, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{metric:30} {corr:>10.3f}")

    # Quintile Analysis
    print("\nQuintile Analysis:")
    print(f"{'Quintile':20} {'Mean CES':>10} {'n':>6} {'% of Total':>12}")
    print("-" * 50)

    total_quintile_n = sum(len(group) for group in stats_summary['quintile_groups'].values())
    for quintile in sorted(stats_summary['quintile_means'].keys()):
        mean = stats_summary['quintile_means'][quintile]
        n = len(stats_summary['quintile_groups'][str(quintile)])
        pct = (n / total_quintile_n) * 100
        print(f"Quintile {int(quintile):15} {mean:>10.2f} {n:>6d} {pct:>11.1f}%")

    if 'quintile_anova' in stats_summary:
        results = stats_summary['quintile_anova']
        print(f"\nANOVA: F={results['f_stat']:.2f}, p={results['p_value']:.4f}")

    # Spend Distribution Analysis
    print("\nSpend Distribution Analysis:")
    print(f"{'Spend Bucket':20} {'Mean CES':>10} {'n':>6} {'% of Total':>12}")
    print("-" * 50)

    bucket_ranges = {
        500: '≤$500',
        1000: '$501-$1,000',
        2000: '$1,001-$2,000',
        3000: '$2,001-$3,000',
        'inf': '>$3,000'
    }

    total_records = sum(stats_summary['bucket_sizes'].values())
    for bucket in sorted([x for x in bucket_ranges.keys()],
                         key=lambda x: float(x) if x != 'inf' else float('inf')):
        n = stats_summary['bucket_sizes'].get(bucket, 0)
        mean = stats_summary['bucket_means'].get(bucket)
        pct = (n / total_records) * 100
        bucket_label = bucket_ranges.get(bucket, str(bucket))

        # Format the mean CES value
        if pd.isna(mean) or n == 0:
            mean_str = "    N/A"
        else:
            mean_str = f"{mean:>10.2f}"

        print(f"{bucket_label:20} {mean_str} {n:>6d} {pct:>11.1f}%")

    if 'bucket_anova' in stats_summary and not pd.isna(stats_summary['bucket_anova']['f_stat']):
        results = stats_summary['bucket_anova']
        print(f"\nANOVA: F={results['f_stat']:.2f}, p={results['p_value']:.4f}")

    # 6. Efficiency Analysis
    # A. Full Population Efficiency Analysis

    if 'population_distribution' in stats_summary:
        print("\nEfficiency Category Analysis (All Ad Spend):")
        print(
            f"Thresholds based on All Ad Spend, including CES Non-Respondents")
        print(f"{'Category':25} {'Mean CES':>10} {'n':>6} {'CES %':>8} {'Pop %':>8}")
        print("-" * 65)

        # Calculate totals
        ces_total = sum(stats['n'] for stats in stats_summary['efficiency_stats'].values())
        pop_total = sum(v for k, v in stats_summary['population_distribution'].items()
                        if k != 'Uncategorized')

        # Define category order
        category_order = [
            'High Spend, High Conv.',
            'High Spend, Low Conv.',
            'Low Spend, High Conv.',
            'Low Spend, Low Conv.'
        ]

        for category in category_order:
            # Get CES stats
            ces_stats = stats_summary['efficiency_stats'].get(category, {})
            ces_mean = ces_stats.get('mean', np.nan)
            ces_n = ces_stats.get('n', 0)
            ces_pct = (ces_n / ces_total * 100) if ces_total > 0 else 0

            # Get population stats
            pop_n = stats_summary['population_distribution'].get(category, 0)
            pop_pct = (pop_n / pop_total * 100) if pop_total > 0 else 0

            # Format mean CES
            mean_str = f"{ces_mean:10.2f}" if not np.isnan(ces_mean) else "      N/A"

            print(f"{category:25} {mean_str} {ces_n:6d} {ces_pct:>8.1f}% {pop_pct:>8.1f}%")

        if 'efficiency_anova' in stats_summary:
            results = stats_summary['efficiency_anova']
            if not np.isnan(results['f_stat']):
                print(f"\nANOVA: F={results['f_stat']:.2f}, p={results['p_value']:.4f}")
            else:
                print("\nANOVA: Insufficient data for analysis")

    # B. CES Respondent-Only Efficiency Analysis
    if 'ces_only_efficiency_stats' in stats_summary:
        print("\nCES Respondent-Only Efficiency Analysis:")
        thresholds = stats_summary['ces_thresholds']
        print(
            f"Thresholds based on CES respondents only (N={sum(stats['n'] for stats in stats_summary['ces_only_efficiency_stats'].values())})")
        print(f"Median Spend: ${thresholds['spend_threshold']:,.2f}, Median Conversions: {thresholds['conv_threshold']:.1f}")
        print()

        print(f"{'Category':25} {'Mean CES':>10} {'n':>6} {'%':>8}")
        print("-" * 55)

        # Calculate total for percentages
        ces_total = sum(stats['n'] for stats in stats_summary['ces_only_efficiency_stats'].values())

        for category in category_order:
            if category in stats_summary['ces_only_efficiency_stats']:
                stats = stats_summary['ces_only_efficiency_stats'][category]
                pct = (stats['n'] / ces_total * 100) if ces_total > 0 else 0
                print(f"{category:25} {stats['mean']:10.2f} {stats['n']:6d} {pct:>8.1f}%")

        if 'ces_only_efficiency_anova' in stats_summary:
            results = stats_summary['ces_only_efficiency_anova']
            if not np.isnan(results['f_stat']):
                print(f"\nANOVA: F={results['f_stat']:.2f}, p={results['p_value']:.4f}")
            else:
                print("\nANOVA: Insufficient data for analysis")

        print("\nDetailed results saved to '../outputs/analyses/ad_spend_correlations.csv'")

def format_category_stats(means_dict, sizes_dict):
    """Format category statistics with means and sample sizes."""
    formatted_stats = {}
    for category in means_dict.keys():
        mean = means_dict[category]
        n = sizes_dict[category]
        formatted_stats[category] = {
            'mean': mean,
            'n': n
        }
    return formatted_stats

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
    # Ad Spend Analysis
    print("\nPerforming ad spend analysis...")
    ad_spend_data, correlations, stats_summary = analyze_ces_vs_ad_spend(ces_data_fe)

    # Log key findings
    print("\nKey Ad Spend Correlations:")
    for metric, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{metric}: {corr:.3f}")

    # Generate visualizations with the analyzed data
    generate_all_visualizations(ces_data_fe, model, config, combinatorial_results)

    # Optional: Save correlations for later reference or reporting
    if correlations:
        correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with CES'])
        correlation_df.to_csv('../outputs/ad_spend_correlations.csv')