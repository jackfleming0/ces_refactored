from itertools import combinations
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def analyze_group_stability(df, group_columns, time_column, ces_column, min_responses):

    # Group by the specified columns and time
    grouped = df.groupby(group_columns + [time_column])
    ces_by_group = grouped.agg({ces_column: ['mean', 'count']}).reset_index()
    ces_by_group.columns = group_columns + [time_column, 'mean_ces', 'response_count']
    ces_by_group_filtered = ces_by_group[ces_by_group['response_count'] >= min_responses]
    pivoted = ces_by_group_filtered.pivot_table(index=group_columns, columns=time_column, values='mean_ces')

    results = []
    for group, group_data in pivoted.iterrows():
        group_data = group_data.dropna()
        if len(group_data) < 2:
            continue

        std_dev, mean = group_data.std(), group_data.mean()
        cv = std_dev / mean if mean != 0 else np.nan
        X = np.arange(len(group_data)).reshape(-1, 1)
        y = group_data.values
        reg = LinearRegression().fit(X, y)
        trend_slope, r_squared = reg.coef_[0], reg.score(X, y)

        total_responses = ces_by_group_filtered[
            ces_by_group_filtered[group_columns].apply(tuple, axis=1) == group
            ]['response_count'].sum()

        result = {col: val for col, val in zip(group_columns, group)}
        result.update({
            'std_dev': std_dev,
            'cv': cv,
            'trend_slope': trend_slope,
            'r_squared': r_squared,
            'data_points': len(group_data),
            'total_responses': total_responses
        })
        results.append(result)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['stability_score'] = (
                results_df['cv'].fillna(0) + abs(results_df['trend_slope'].fillna(0)) + (
                    1 - results_df['r_squared'].fillna(0))
        )
        return results_df.sort_values('stability_score')
    else:
        return pd.DataFrame(
            columns=group_columns + ['std_dev', 'cv', 'trend_slope', 'r_squared', 'data_points', 'total_responses',
                                     'stability_score'])


def generate_column_combinations(columns, min_combo, max_combo):
    """
    Generate combinations of columns, limited to at most two additional factors.

    Parameters:
    columns (list): List of column names
    min_combo (int): Minimum number of columns in a combination
    max_combo (int): Maximum number of columns in a combination

    Returns:
    list: List of column combinations
    """
    all_combinations = []
    for r in range(min_combo, min(max_combo, len(columns)) + 1):
        all_combinations.extend(combinations(columns, r))

    return all_combinations


def perform_combinatorial_analysis(df, base_columns, additional_columns, time_column, ces_column, min_responses,
                                   min_combo, max_combo):
    column_combinations = generate_column_combinations(additional_columns, min_combo, max_combo)
    results = {}
    for combo in column_combinations:
        group_columns = base_columns + list(combo)
        combo_name = " + ".join(group_columns)

        # Debugging: Ensure 'db_number' and other columns exist in the DataFrame
        print(f"Analyzing combination: {combo_name}")
        if 'db_number' in group_columns:
            print(f"Unique values in db_number: {df['db_number'].unique()}")

        # Double-check for NaN values in db_number and drop them
        if 'db_number' in group_columns:
            df = df.dropna(subset=['db_number'])

        # Optionally, make a copy of the DataFrame to avoid issues with view vs copy
        df = df.copy()

        # Call the stability analysis function
        stability_results = analyze_group_stability(df, group_columns, time_column, ces_column, min_responses)
        results[combo_name] = stability_results

    return results


def identify_top_stable_elastic_groups(combinatorial_results, n_groups=5):
    """
    Identify the most stable and most elastic groups across all combinations.

    Parameters:
    combinatorial_results (dict): The output from perform_combinatorial_analysis function
    n_groups (int): Number of groups to identify in each category

    Returns:
    tuple: Two dataframes containing the most stable and most elastic groups
    """
    # Concatenate the results from all combinations into a single DataFrame
    all_results = pd.concat(combinatorial_results.values(), keys=combinatorial_results.keys())
    all_results = all_results.reset_index(level=1, drop=True).reset_index(names='combination')

    # Sort by stability score to get the most stable and most elastic groups
    most_stable = all_results.nsmallest(n_groups, 'stability_score')
    most_elastic = all_results.nlargest(n_groups, 'stability_score')

    return most_stable, most_elastic