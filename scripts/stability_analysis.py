from itertools import combinations
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx


def enhanced_analyze_group_stability(df, group_columns, time_column, ces_column, min_responses):
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

        # Calculate cumulative last CES
        cumulative_data = ces_by_group_filtered[
            ces_by_group_filtered[group_columns].apply(tuple, axis=1) == group
        ]
        cumulative_last_ces = (cumulative_data['mean_ces'] * cumulative_data['response_count']).sum() / cumulative_data['response_count'].sum()

        result = {col: val for col, val in zip(group_columns, group)}
        result.update({
            'std_dev': std_dev,
            'cv': cv,
            'trend_slope': trend_slope,
            'r_squared': r_squared,
            'data_points': len(group_data),
            'total_responses': total_responses,
            'mean_ces': mean,
            'first_ces': group_data.iloc[0],
            'last_ces': group_data.iloc[-1],
            'cumulative_last_ces': cumulative_last_ces,
            'ces_change': group_data.iloc[-1] - group_data.iloc[0],
            'cumulative_ces_change': cumulative_last_ces - group_data.iloc[0]
        })
        results.append(result)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df['stability_score'] = calculate_enhanced_stability_score(results_df)
        results_df['trend_direction'] = np.where(results_df['trend_slope'] > 0, 'Improving', 'Declining')
        results_df['performance_category'] = categorize_performance(results_df)
        return results_df.sort_values('stability_score')
    else:
        return pd.DataFrame(columns=group_columns + ['std_dev', 'cv', 'trend_slope', 'r_squared', 'data_points',
                                                     'total_responses', 'mean_ces', 'first_ces', 'last_ces',
                                                     'cumulative_last_ces', 'ces_change', 'cumulative_ces_change',
                                                     'stability_score', 'trend_direction', 'performance_category'])



def calculate_enhanced_stability_score(df):
    # Normalize components to 0-1 range
    cv_norm = (df['cv'] - df['cv'].min()) / (df['cv'].max() - df['cv'].min())
    trend_norm = (df['trend_slope'].abs() - df['trend_slope'].abs().min()) / (
                df['trend_slope'].abs().max() - df['trend_slope'].abs().min())
    r_squared_norm = 1 - df['r_squared']  # Already in 0-1 range

    # Calculate stability score with weights
    return 0.4 * cv_norm + 0.3 * trend_norm + 0.3 * r_squared_norm


def categorize_performance(df):
    conditions = [
        (df['mean_ces'] > df['mean_ces'].mean()) & (df['trend_slope'] > 0),
        (df['mean_ces'] > df['mean_ces'].mean()) & (df['trend_slope'] <= 0),
        (df['mean_ces'] <= df['mean_ces'].mean()) & (df['trend_slope'] > 0),
        (df['mean_ces'] <= df['mean_ces'].mean()) & (df['trend_slope'] <= 0)
    ]
    choices = ['High Performing', 'At Risk', 'Improving', 'Needs Attention']
    return np.select(conditions, choices, default='Uncategorized')

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
        stability_results = enhanced_analyze_group_stability(df, group_columns, time_column, ces_column, min_responses)
        results[combo_name] = stability_results

        # Call the functions to generate the visualizations
        #plot_heatmap(results)
        #plot_parallel_coordinates(results)
        #plot_network_graph(results)
        plot_bubble_chart(results)
        #plot_treemap(results)
        #plot_radar_chart(results)

    return results


def identify_top_stable_elastic_groups(combinatorial_results, n_groups=50):
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


# 1. Heatmap of Stability Scores
def plot_heatmap(results):
    # Prepare data for heatmap
    heatmap_data = []
    for combo, data in results.items():
        for _, row in data.iterrows():
            heatmap_data.append({
                'Combination': combo,
                'Group': ' '.join([f"{col}:{row[col]}" for col in data.columns if
                                   col not in ['std_dev', 'cv', 'trend_slope', 'r_squared', 'data_points',
                                               'total_responses', 'stability_score']]),
                'Stability Score': row['stability_score']
            })
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_pivot = heatmap_df.pivot(index='Combination', columns='Group', values='Stability Score')

    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_pivot, cmap='YlOrRd_r', annot=True, fmt='.2f', cbar_kws={'label': 'Stability Score'})
    plt.title('Heatmap of Stability Scores')
    plt.tight_layout()
    plt.show()


# 2. Parallel Coordinates Plot
def plot_parallel_coordinates(results):
    # Prepare data for parallel coordinates
    pc_data = pd.concat(results.values()).reset_index(drop=True)
    pc_data['Combination'] = pc_data.apply(lambda row: ' '.join([f"{col}:{row[col]}" for col in pc_data.columns if
                                                                 col not in ['std_dev', 'cv', 'trend_slope',
                                                                             'r_squared', 'data_points',
                                                                             'total_responses', 'stability_score']]),
                                           axis=1)

    fig = go.Figure(data=
    go.Parcoords(
        line=dict(color=pc_data['stability_score'],
                  colorscale='Viridis',
                  showscale=True),
        dimensions=list([
            dict(label='CV', values=pc_data['cv']),
            dict(label='Trend Slope', values=pc_data['trend_slope']),
            dict(label='R-squared', values=pc_data['r_squared']),
            dict(label='Stability Score', values=pc_data['stability_score'])
        ])
    )
    )
    fig.update_layout(title='Parallel Coordinates Plot of Combinatorial Analysis Results')
    fig.show()


# 3. Network Graph
def plot_network_graph(results):
    G = nx.Graph()
    for combo, data in results.items():
        factors = combo.split(' + ')
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                if G.has_edge(factors[i], factors[j]):
                    G[factors[i]][factors[j]]['weight'] += 1
                else:
                    G.add_edge(factors[i], factors[j], weight=1)

    pos = nx.spring_layout(G)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title('Network Graph of Factor Interactions')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# 4. Bubble Chart
def plot_bubble_chart(results):
    # Prepare data for bubble chart
    factor_data = []
    for combo, data in results.items():
        factors = combo.split(' + ')
        for factor in factors:
            factor_data.append({
                'Factor': factor,
                'Avg Stability Score': data['stability_score'].mean(),
                'Frequency': 1,
                'Avg CV': data['cv'].mean()
            })

    bubble_df = pd.DataFrame(factor_data).groupby('Factor').agg({
        'Avg Stability Score': 'mean',
        'Frequency': 'sum',
        'Avg CV': 'mean'
    }).reset_index()

    fig = px.scatter(bubble_df, x='Factor', y='Avg Stability Score',
                     size='Frequency', color='Avg CV',
                     hover_name='Factor', size_max=60,
                     title='Bubble Chart of Stability Drivers')
    fig.show()


# 5. Treemap
def plot_treemap(results):
    # Prepare data for treemap
    treemap_data = []
    for combo, data in results.items():
        primary_factor = combo.split(' + ')[0]
        for _, row in data.iterrows():
            treemap_data.append({
                'Primary Factor': primary_factor,
                'Combination': combo,
                'Total Responses': max(row['total_responses'], 1),  # Ensure at least 1 response
                'Stability Score': row['stability_score']
            })

    treemap_df = pd.DataFrame(treemap_data)

    # Filter out rows with zero Total Responses
    treemap_df = treemap_df[treemap_df['Total Responses'] > 0]

    if treemap_df.empty:
        print("No data available for treemap after filtering zero responses.")
        return

    fig = px.treemap(treemap_df,
                     path=['Primary Factor', 'Combination'],
                     values='Total Responses',
                     color='Stability Score',
                     color_continuous_scale='RdYlGn_r',
                     title='Treemap of Stable Combinations')
    fig.show()


# 6. Radar Chart
def plot_radar_chart(results):
    # Prepare data for radar chart
    all_results = pd.concat(results.values()).reset_index(drop=True)
    top_stable = all_results.nsmallest(5, 'stability_score')
    top_elastic = all_results.nlargest(5, 'stability_score')
    radar_data = pd.concat([top_stable, top_elastic])

    metrics = ['cv', 'trend_slope', 'r_squared', 'stability_score']

    fig = go.Figure()

    for _, row in radar_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=metrics,
            fill='toself',
            name=f"{row['ClientUser_Type']} ({'Stable' if row['stability_score'] < 1 else 'Elastic'})"
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(radar_data[metrics].max())])),
        showlegend=True,
        title='Radar Chart of Key Metrics for Top Stable and Elastic Groups'
    )
    fig.show()
