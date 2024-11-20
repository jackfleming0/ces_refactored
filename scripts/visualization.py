import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import math
from stability_analysis import identify_top_stable_elastic_groups
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import logging
from scipy import stats as scipy_stats
import time




# Helper function to always set y-axis from 0 to 7
def set_ces_y_axis(ax):
    ax.set_ylim(0, 7)


def plot_ces_scores_by_response_group(df, group_column='Response_Group', target_column='CES_Response_Value'):
    # Calculate the mean CES score per response group
    mean_ces_by_group = df.groupby(group_column)[target_column].mean()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=mean_ces_by_group.index, y=mean_ces_by_group.values, palette='viridis')

    # Add labels to the bars
    for i, value in enumerate(mean_ces_by_group.values):
        ax.text(i, value + 0.05, f'{value:.2f}', ha='center', va='bottom', color='black', fontweight='bold')

    # Set the y-axis to always range from 1 to 7
    ax.set_ylim(0, 7)

    # Add labels and title
    plt.title('Mean CES Scores by Response Group')
    plt.xlabel('Response Group')
    plt.ylabel('Mean CES Score')

    # Ensure the layout is tight
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_ces_by_ntile(df, column, n, target='CES_Response_Value'):
    df[f'{column}_decile'] = pd.qcut(df[column], n)
    mean_ces_by_decile = df.groupby(f'{column}_decile')[target].mean()
    overall_mean = df[target].mean()

    plt.figure(figsize=(10, 6))
    ax = mean_ces_by_decile.plot(kind='bar')

    # Adding value labels to each bar
    for i, v in enumerate(mean_ces_by_decile):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center')

    # Adding horizontal line for overall mean
    plt.axhline(overall_mean, color='red', linestyle='--', label=f'Mean CES: {overall_mean:.2f}')
    plt.title(f'CES by {column.capitalize()} Ranges')
    plt.xlabel(f'{column.capitalize()} Ranges')
    plt.ylabel('Mean CES Score')
    plt.xticks(rotation=45)
    set_ces_y_axis(ax)  # Set y-axis to always run 0-7
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_combined_ces_distribution(df, ces_column='CES_Response_Value'):
    plt.figure(figsize=(14, 10))  # Increased figure size to accommodate legends

    # Drop any NaN values in relevant columns
    df = df.dropna(subset=['Response_Group', 'ClientUser_Type'])

    # Convert Response_Group and ClientUser_Type to categorical if necessary
    df['Response_Group'] = df['Response_Group'].astype('category')
    df['ClientUser_Type'] = df['ClientUser_Type'].astype('category')

    # Create the first distribution plot for Response_Group
    ax1 = plt.subplot(211)
    sns.kdeplot(
        data=df,
        x=ces_column,
        hue='Response_Group',
        fill=True,
        common_norm=False,
        alpha=.5,
        linewidth=2,
        ax=ax1
    )

    ax1.set_title('CES Distribution by Response Group')
    ax1.set_xlabel('')  # Remove x-label from the top subplot
    ax1.set_xlim(1, 7)  # Set x-axis limits to the CES scale (assuming 1-7)

    # Manually set legend for the first plot
    handles, labels = ax1.get_legend_handles_labels()  # Get the handles and labels from the plot
    ax1.legend(handles=handles, labels=labels, title='Response Group', loc='upper right', bbox_to_anchor=(1.15, 1))

    # Create the second distribution plot for ClientUser_Type
    ax2 = plt.subplot(212, sharex=ax1)  # Share x-axis with the first subplot
    sns.kdeplot(
        data=df,
        x=ces_column,
        hue='ClientUser_Type',
        fill=True,
        common_norm=False,
        alpha=.5,
        linewidth=2,
        ax=ax2
    )

    ax2.set_title('CES Distribution by Client User Type')
    ax2.set_xlabel('CES Response Value')
    ax2.set_xlim(1, 7)  # Set x-axis limits to the CES scale (assuming 1-7)

    # Manually set legend for the second plot
    handles, labels = ax2.get_legend_handles_labels()  # Get the handles and labels from the plot
    ax2.legend(handles=handles, labels=labels, title='Client User Type', loc='upper right', bbox_to_anchor=(1.15, 1))

    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()

def analyze_ces_distributions(df):
    plot_combined_ces_distribution(df)

    # Add summary statistics
    print("\nSummary Statistics:")
    print("\nBy Response Group:")
    print(df.groupby('Response_Group')['CES_Response_Value'].describe())

    print("\nBy Client User Type:")
    print(df.groupby('ClientUser_Type')['CES_Response_Value'].describe())


def plot_ces_distribution_by_group(df, group_column, response_column='CES_Response'):
    # Define the custom order
    custom_order = ["Very Difficult", "Difficult", "Somewhat Difficult", "Neutral",
                    "Somewhat Easy", "Easy", "Very Easy"]

    # Create the crosstab
    grouped_data = pd.crosstab(df[group_column], df[response_column], normalize='index') * 100

    # Reorder the columns according to the custom order
    grouped_data = grouped_data.reindex(columns=custom_order)

    # Create a color map that matches the order of the stacks
    colors = plt.cm.viridis(np.linspace(0, 1, len(custom_order)))

    # Plot the stacked bar chart
    ax = grouped_data.plot(kind='bar', stacked=True, color=colors, figsize=(10, 6))

    plt.ylabel('Percentage')
    plt.title(f'Percentage Distribution of {response_column} by {group_column}')
    plt.xticks(rotation=0)
    plt.legend(title=response_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_ces_trend_over_time(df, target_column='CES_Response_Value', group_column='Response_Group'):
    trend = df.groupby(group_column)[target_column].mean()

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=trend, marker='o')
    plt.title('CES Trend Over Time by Response Group')
    plt.xlabel('Response Group')
    plt.ylabel('Mean CES Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_ces_by_category(df, category_column, target_column='CES_Response_Value'):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=category_column, y=target_column, data=df)
    plt.title(f'CES by {category_column}')
    plt.xlabel(category_column)
    plt.ylabel('CES Score')
    set_ces_y_axis(ax)  # Set y-axis to always run 0-7
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_ces_distribution(df, target_column='CES_Response_Value'):
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df[target_column], bins=7, kde=True)
    plt.title('Distribution of CES Responses')
    plt.xlabel('CES Response Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def plot_ces_by_time_of_day(df, time_column='Response_Timestamp', target_column='CES_Response_Value'):
    df['hour'] = pd.to_datetime(df[time_column]).dt.hour
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='hour', y=target_column, data=df)
    plt.title('CES by Time of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('CES Score')
    set_ces_y_axis(ax)  # Set y-axis to always run 0-7
    plt.tight_layout()
    plt.show()


def plot_ces_by_day_of_week(df, time_column='Response_Timestamp', target_column='CES_Response_Value'):
    df['day_of_week'] = pd.to_datetime(df[time_column]).dt.day_name()
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='day_of_week', y=target_column, data=df,
                     order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.title('CES by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('CES Score')
    set_ces_y_axis(ax)  # Set y-axis to always run 0-7
    plt.tight_layout()
    plt.show()


def plot_feature_importances(model, X):
    feature_importances = model.feature_importances_
    features = X.columns
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(features[sorted_idx], feature_importances[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances from Random Forest')
    plt.tight_layout()
    plt.show()

    # SHAP summary plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)


def plot_correlation_with_ces(df, target_column='CES_Response_Value'):
    correlation_matrix = df.corr()[target_column].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    correlation_matrix.dropna().plot(kind='bar')
    plt.title(f'Correlation of Features with {target_column}')
    plt.xlabel('Feature')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_mean_ces_by_db(df, db_column='db_number', target_column='CES_Response_Value'):
    df_mean = df.groupby(db_column)[target_column].mean()
    df_count = df.groupby(db_column)[target_column].count()

    fig, ax1 = plt.subplots(figsize=(12, 6))  # Increased figure width for better label visibility
    ax1.bar(df_mean.index, df_count, color='lightblue', label='Count')
    ax2 = ax1.twinx()
    line = ax2.plot(df_mean.index, df_mean, color='orange', label='Mean CES', marker='o')

    # Add CES value labels above each dot
    for x, y in zip(df_mean.index, df_mean):
        ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    ax1.set_xlabel('DB Number')
    ax1.set_ylabel('Count')
    ax2.set_ylabel('Mean CES')
    set_ces_y_axis(ax2)  # Set y-axis to always run 0-7

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Mean CES and Response Count by DB Number')
    plt.tight_layout()
    plt.show()

def plot_boxplot_by_category(df, category_column, target_column='CES_Response_Value'):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=category_column, y=target_column, data=df)

    # Calculate means for each category
    means = df.groupby(category_column)[target_column].mean()

    # Add mean labels
    for i, mean_val in enumerate(means):
        ax.text(i, mean_val, f'Mean: {mean_val:.2f}',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontweight='bold',
                color='yellow')

    # Add scatter plot for individual data points
    sns.stripplot(x=category_column, y=target_column, data=df,
                  size=4, color=".3", alpha=0.6, ax=ax)

    plt.title(f'CES by {category_column}')
    plt.xlabel(category_column)
    plt.ylabel('CES Score')
    set_ces_y_axis(ax)  # Set y-axis to always run 0-7
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_elasticity_by_cohort(df, cohort_column='Response_Group', target_column='CES_Response_Value'):
    group_stats = df.groupby(cohort_column)[target_column].agg(['std', 'mean'])
    group_stats['cv'] = group_stats['std'] / group_stats['mean']

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=group_stats.index, y='cv', data=group_stats)
    plt.title('Elasticity (CV) by Cohort')
    plt.xlabel('Cohort')
    plt.ylabel('Coefficient of Variation')
    set_ces_y_axis(ax)  # Set y-axis to always run 0-7
    plt.tight_layout()
    plt.show()


def visualize_combinatorial_results(combinatorial_results):
    n_combos = len(combinatorial_results)
    n_cols = min(3, n_combos)
    n_rows = math.ceil(n_combos / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_combos > 1 else [axes]

    for i, (combo_name, results) in enumerate(combinatorial_results.items()):
        ax = axes[i]
        sns.scatterplot(x='cv', y='trend_slope', size='data_points', hue='stability_score', data=results, ax=ax)
        ax.set_title(combo_name, fontsize=10)
        ax.set_xlabel('Coefficient of Variation')
        ax.set_ylabel('Trend Slope')

    plt.tight_layout()
    plt.show()


def plot_ces_radar_chart(df, categories, target_column='CES_Response_Value'):
    # Calculate mean CES and count for each category
    grouped = df.groupby(categories).agg({target_column: 'mean', categories[0]: 'count'})
    grouped.columns = ['mean', 'count']

    # Ensure we have a value for each category
    index_values = grouped.index.tolist()
    values = grouped['mean'].tolist()
    counts = grouped['count'].tolist()

    # Number of variables
    num_vars = len(index_values)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]

    # Close the polygon by appending the start value to the end
    values += values[:1]
    angles += angles[:1]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

    # Plot data
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # Set category labels with n-size
    ax.set_xticks(angles[:-1])
    labels = [f"{' - '.join(map(str, cat))}\n(n={count})" for cat, count in zip(index_values, counts)]
    ax.set_xticklabels(labels, wrap=True)

    # Set y-axis to always range from 0 to 7
    ax.set_ylim(0, 7)
    ax.set_yticks(np.arange(1, 8))

    # Add dynamic title using original category names
    category_names = ' & '.join(categories)
    plt.title(f'CES Radar Chart by {category_names}', size=16, y=1.1)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_ces_sankey_diagram(df, source_col, target_col, value_col='CES_Response_Value'):
    # Create a dataframe with the flow
    flow_df = df.groupby([source_col, target_col]).size().reset_index(name='count')

    # Get unique labels
    all_labels = pd.concat([flow_df[source_col], flow_df[target_col]]).unique()

    # Create a mapping of labels to indices
    label_to_index = {label: index for index, label in enumerate(all_labels)}

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color="blue"
        ),
        link=dict(
            source=[label_to_index[src] for src in flow_df[source_col]],
            target=[label_to_index[tgt] for tgt in flow_df[target_col]],
            value=flow_df['count']
        ))])

    # Update the layout
    fig.update_layout(title_text=f"CES Flow: {source_col} to {target_col}", font_size=10)

    # Show the plot
    fig.show()


def plot_filtered_ces_distribution(df, ces_column='CES_Response_Value'):
    # Filter data for ClientUser_Type == 'Primary Manager'
    filtered_df = df[df['ClientUser_Type'] == 'Primary Manager']

    # Check if filtered dataframe is not empty
    if filtered_df.empty:
        print("No data available for ClientUser_Type = 'Primary Manager'")
        return

    plt.figure(figsize=(10, 6))

    # Create the distribution plot for ResponseGroup, overlaid on the same chart
    sns.kdeplot(
        data=filtered_df,
        x=ces_column,
        hue='Response_Group',
        fill=True,
        common_norm=False,
        alpha=.5,
        linewidth=2
    )

    # Set the title and axis labels
    plt.title('CES Distribution for Primary Manager by Response Group')
    plt.xlabel('CES Response Value')
    plt.xlim(1, 7)  # Assuming CES is on a scale of 1-7

    # Manually set the legend
    handles, labels = plt.gca().get_legend_handles_labels()  # Get handles and labels
    response_group_labels = filtered_df['Response_Group'].unique()  # Extract unique ResponseGroup values
    plt.legend(handles=handles, labels=response_group_labels, title='Response Group', loc='upper right')

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_ces_by_variable_and_response_group(df, variable_column, title=None):
    # Get all unique response groups and sort them
    group_order = sorted(df['Response_Group'].unique(),
                        key=lambda x: int(x.split()[-1]))

    # Calculate mean CES for each combination
    grouped_data = df.groupby([variable_column, 'Response_Group'])['CES_Response_Value'].mean().unstack()

    # Reorder columns based on the sorted order
    grouped_data = grouped_data.reindex(columns=group_order)

    # Create the line plot
    plt.figure(figsize=(12, 6))
    for value in grouped_data.index:
        plt.plot(grouped_data.columns, grouped_data.loc[value], marker='o', label=value)

    # Customize the plot
    if title is None:
        title = f'Mean CES by {variable_column} and Response Group'
    plt.title(title)
    plt.xlabel('Response Group')
    plt.ylabel('Mean CES Score')
    plt.ylim(1, 7)  # Set y-axis range from 1 to 7
    plt.legend(title=variable_column)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()
    time.sleep(3)  # Add delay
    plt.close()


def plot_regional_ad_spend_analysis(df, save_path=None):
    """
    Create visualization showing regional patterns in ad spend and CES.

    Parameters:
    df (pandas.DataFrame): DataFrame with CES and ad spend features
    save_path (str, optional): Path to save the visualization
    """
    # Create subplot grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Scatter plot of regional average CES vs average ad spend
    regional_data = df.groupby('MLS_Region').agg({
        'CES_Response_Value': 'mean',
        'total_monthly_spend': 'mean',
        'cost_per_lead': 'mean'
    }).reset_index()

    sns.scatterplot(x='total_monthly_spend', y='CES_Response_Value',
                    size='cost_per_lead', data=regional_data, ax=ax1)
    ax1.set_title('Regional Average CES vs Ad Spend')
    ax1.set_xlabel('Average Monthly Spend')
    ax1.set_ylabel('Average CES Score')
    set_ces_y_axis(ax1)

    # 2. Box plot of CES by region (top N regions by response count)
    top_regions = df['MLS_Region'].value_counts().nlargest(10).index
    region_data = df[df['MLS_Region'].isin(top_regions)]

    sns.boxplot(x='MLS_Region', y='CES_Response_Value', data=region_data, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_title('CES Distribution by Top Regions')
    ax2.set_xlabel('MLS Region')
    ax2.set_ylabel('CES Score')
    set_ces_y_axis(ax2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
    time.sleep(3)  # Add a 3-second delay. this helped to prevent getting rate limited
    plt.close()

def generate_ad_spend_visualizations(df):
    """Generate comprehensive visualizations for ad spend analysis."""
    print("Generating ad spend visualizations...")

    # Start with the overall comparison
    print("\nGenerating ad spend vs no ad spend comparison...")
    plot_ad_spend_comparison(df)

    # Only analyze records with ad spend
    ad_spend_data = df[df['AdSpendYN']].copy()

    if len(ad_spend_data) == 0:
        print("No ad spend data available for visualization")
        return

    # 1. Correlation Analysis Plot
    plt.figure(figsize=(12, 6))
    correlations = {
        'AdSpendCTR': 0.129,
        'AdSpendSearchImprShare': 0.088,
        'AdSpendConversions': 0.057,
        'AdSpendCost': 0.044,
        'ConversionsPerSeat': 0.017,
        'AdSpendCPC': -0.008,
        'SpendPerSeat': -0.006,
        'AdSpendCostPerConversion': -0.003
    }

    # Sort by absolute value
    sorted_correlations = dict(sorted(correlations.items(),
                                      key=lambda x: abs(x[1]),
                                      reverse=True))

    ax = sns.barplot(x=list(sorted_correlations.values()),
                     y=list(sorted_correlations.keys()),
                     palette='viridis')

    # Add value labels
    for i, v in enumerate(sorted_correlations.values()):
        ax.text(v + (0.01 if v >= 0 else -0.01),
                i,
                f'{v:.3f}',
                va='center',
                fontweight='bold')

    plt.title('Ad Spend Metrics Correlation with CES')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Efficiency Categories Analysis
    plt.figure(figsize=(12, 6))
    efficiency_data = {
        'High Spend, High Conv.': {'mean': 5.36, 'n': 101},
        'High Spend, Low Conv.': {'mean': 5.44, 'n': 25},
        'Low Spend, High Conv.': {'mean': 5.57, 'n': 23},
        'Low Spend, Low Conv.': {'mean': 5.12, 'n': 60}
    }

    means = [v['mean'] for v in efficiency_data.values()]
    categories = list(efficiency_data.keys())

    ax = sns.barplot(x=categories, y=means, palette='viridis')

    # Add both sample size and mean value annotations
    for i, (cat, data) in enumerate(efficiency_data.items()):
        # Add mean value at top of bar
        ax.text(i, data['mean'], f'CES: {data["mean"]:.2f}',
                ha='center', va='bottom', fontweight='bold')
        # Add sample size below category label
        ax.text(i, -0.1, f'n={data["n"]}',
                ha='center', va='top', color='darkblue')

    plt.title('CES by Efficiency Category')
    plt.xlabel('Category')
    plt.ylabel('Mean CES Score')
    set_ces_y_axis(ax)
    # Adjust bottom margin to accommodate sample size labels
    plt.margins(y=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Spend Distribution Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    spend_data = {
        '>$3,000': {'pct': 58.9, 'ces': 5.36},
        '$2,001-$3,000': {'pct': 20.6, 'ces': 5.44},
        '$1,001-$2,000': {'pct': 5.3, 'ces': 5.00},
        '≤$500': {'pct': 15.3, 'ces': 5.12}
    }

    # Pie chart
    plt.sca(ax1)
    patches, texts, autotexts = plt.pie(
        [v['pct'] for v in spend_data.values()],
        labels=[f"{k}\n({v['pct']}%)" for k, v in spend_data.items()],
        autopct='%1.1f%%',
        colors=sns.color_palette('viridis', n_colors=len(spend_data)))

    # Make percentage labels more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    plt.title('Distribution of Ad Spend')

    # CES by spend level
    plt.sca(ax2)
    bars = sns.barplot(x=list(spend_data.keys()),
                       y=[v['ces'] for v in spend_data.values()],
                       palette='viridis')

    # Add value labels on bars
    for i, v in enumerate(spend_data.values()):
        bars.text(i, v['ces'], f'CES: {v["ces"]:.2f}',
                  ha='center', va='bottom', fontweight='bold')

    plt.title('Mean CES by Spend Level')
    set_ces_y_axis(ax2)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # 4. Quintile Analysis
    if 'AdSpend_Quintile' in ad_spend_data.columns:
        plt.figure(figsize=(10, 6))

        # Calculate means for each quintile for labels
        quintile_means = ad_spend_data.groupby('AdSpend_Quintile')['CES_Response_Value'].agg(
            ['mean', 'count']).reset_index()

        ax = sns.boxplot(x='AdSpend_Quintile',
                         y='CES_Response_Value',
                         data=ad_spend_data,
                         palette='viridis')

        # Add mean and count labels
        for i, row in quintile_means.iterrows():
            ax.text(i, row['mean'], f'Mean: {row["mean"]:.2f}\nn={row["count"]}',
                    ha='center', va='bottom', fontweight='bold')

        plt.title('CES Distribution by Ad Spend Quintile')
        set_ces_y_axis(ax)
        plt.tight_layout()
        plt.show()

    print("Ad spend visualizations completed.")


def plot_database_performance_comparison(df, db_column='db_cohort'):
    """
    Create a detailed visualization of CES performance across database categories.

    Parameters:
    df (pandas.DataFrame): DataFrame containing CES and database information
    db_column (str): Column name containing database cohort information
    """
    print("Starting database performance comparison...")
    plt.close('all')  # Close any existing plots
    time.sleep(1)  # Initial delay

    # Calculate statistics from the data
    db_stats = (df.groupby(db_column)['CES_Response_Value']
                .agg(['mean', 'count', 'std'])
                .round(3))

    # Perform statistical tests
    from scipy import stats as scipy_stats  # Renamed to avoid conflict
    # ANOVA test
    categories = df[db_column].unique()
    groups = [df[df[db_column] == cat]['CES_Response_Value'] for cat in categories]
    f_stat, anova_p = scipy_stats.f_oneway(*groups)

    # Pairwise t-tests between refactored groups
    refactored_current = df[df[db_column] == 'refactored_current']['CES_Response_Value']
    refactored_legacy = df[df[db_column] == 'refactored_legacy']['CES_Response_Value']
    t_stat, p_value = scipy_stats.ttest_ind(refactored_current, refactored_legacy)

    # Create significance dictionary
    significance = {
        cat: p_value < 0.05 if cat in ['refactored_current', 'refactored_legacy'] else False
        for cat in categories
    }

    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Create bars
    bars = plt.bar(range(len(db_stats)), db_stats['mean'], width=0.6)

    # Color the bars based on significance
    for i, bar in enumerate(bars):
        category = db_stats.index[i]
        bar.set_color('#3B82F6' if significance[category] else '#93C5FD')

        # Add mean value label
        plt.text(i, db_stats.loc[db_stats.index[i], 'mean'],
                 f"{db_stats.loc[db_stats.index[i], 'mean']:.2f}",
                 ha='center', va='bottom',
                 fontweight='bold')

        # Add sample size
        plt.text(i, 0.5, f"n={int(db_stats.loc[db_stats.index[i], 'count'])}",
                 ha='center', va='bottom',
                 color='darkblue',
                 fontweight='bold')

        # Add significance indicator
        if significance[category]:
            plt.text(i, 0.2, '★ Significant',
                     ha='center', va='bottom',
                     color='#2563EB',
                     fontweight='bold')

    # Customize the plot
    plt.title('Database Performance Comparison',
              pad=20, fontsize=14, fontweight='bold')

    plt.xticks(range(len(db_stats)), db_stats.index,
               rotation=45, ha='right')

    set_ces_y_axis(ax)
    plt.ylabel('Mean CES Score')

    # Add statistical annotations
    stat_text = (f"ANOVA: F={f_stat:.2f}, p={anova_p:.4f}\n"
                 f"Refactored Comparison: p={p_value:.4f}")

    plt.text(0.5, -0.2, stat_text,
             ha='center', va='center',
             transform=ax.transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.show()
    time.sleep(3)  # Add a 3-second delay. this helped to prevent getting rate limited
    plt.close()

def plot_detailed_db_analysis(df, db_column='db_cohort',
                              time_column='Response_Group',
                              ces_column='CES_Response_Value'):
    """
    Create a multi-panel analysis of database performance.

    Parameters:
    df (pandas.DataFrame): DataFrame containing CES and database information
    db_column (str): Column name containing database cohort information
    time_column (str): Column name containing time/group information
    ces_column (str): Column name containing CES scores
    """
    print("Starting detailed db analysis...")  # Add debug print

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Main comparison (top left)
    sns.boxplot(data=df, x=db_column, y=ces_column, ax=ax1)
    ax1.set_title('CES Distribution by Database Category')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    set_ces_y_axis(ax1)

    # 2. Trend over time (top right)
    sns.lineplot(data=df, x=time_column, y=ces_column,
                 hue=db_column, ax=ax2, marker='o')
    ax2.set_title('CES Trends by Database Category')
    set_ces_y_axis(ax2)

    # 3. Response distribution (bottom left)
    sns.kdeplot(data=df, x=ces_column, hue=db_column, ax=ax3)
    ax3.set_title('CES Score Distribution by Database Category')

    # 4. Statistical summary (bottom right)
    ax4.clear()
    ax4.axis('off')

    # Calculate statistics
    stats_df = df.groupby(db_column)[ces_column].agg(['count', 'mean', 'std']).round(3)

    # Perform ANOVA
    categories = df[db_column].unique()
    groups = [df[df[db_column] == cat][ces_column] for cat in categories]
    f_stat, anova_p = scipy_stats.f_oneway(*groups)  # Changed to scipy_stats

    # Calculate pairwise t-tests
    ref_current = df[df[db_column] == 'refactored_current'][ces_column]
    ref_legacy = df[df[db_column] == 'refactored_legacy'][ces_column]
    t_stat, p_value = scipy_stats.ttest_ind(ref_current, ref_legacy)  # Changed to scipy_stats

    stats_text = f"""
    Statistical Summary:

    Sample Sizes:
    {stats_df['count'].to_string()}

    Mean Scores:
    {stats_df['mean'].to_string()}

    Standard Deviations:
    {stats_df['std'].to_string()}

    ANOVA Results:
    • F-statistic: {f_stat:.2f}
    • p-value: {anova_p:.4f} {'**' if anova_p < 0.01 else '*' if anova_p < 0.05 else ''}

    Refactored Current vs Legacy:
    • Difference: {ref_current.mean() - ref_legacy.mean():.4f}
    • p-value: {p_value:.4f} {'**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}
    """

    ax4.text(0.1, 0.9, stats_text, fontsize=10,
             verticalalignment='top',
             fontfamily='monospace')

    plt.tight_layout()
    plt.show()
    time.sleep(3)  # Add a 3-second delay. this helped to prevent getting rate limited
    plt.close()

def plot_ad_spend_comparison(df):
    """Compare CES scores between clients with and without ad spend."""
    # Create comparison groups
    with_ad_spend = df[df['AdSpendYN']]['CES_Response_Value']
    without_ad_spend = df[~df['AdSpendYN']]['CES_Response_Value']

    # Calculate statistics
    stats = {
        'With Ad Spend': {
            'mean': with_ad_spend.mean(),
            'std': with_ad_spend.std(),
            'n': len(with_ad_spend),
            'data': with_ad_spend
        },
        'Without Ad Spend': {
            'mean': without_ad_spend.mean(),
            'std': without_ad_spend.std(),
            'n': len(without_ad_spend),
            'data': without_ad_spend
        }
    }

    # Create visualization
    plt.figure(figsize=(12, 6))

    # Create side-by-side box plots with individual points
    ax = plt.gca()

    # Box plots
    sns.boxplot(x=df['AdSpendYN'].map({True: 'With Ad Spend', False: 'Without Ad Spend'}),
                y='CES_Response_Value', data=df,
                palette='viridis',
                width=0.5)

    # Add individual points
    sns.stripplot(x=df['AdSpendYN'].map({True: 'With Ad Spend', False: 'Without Ad Spend'}),
                  y='CES_Response_Value', data=df,
                  size=4, color=".3", alpha=0.3)

    # Add detailed statistics annotations
    for i, (label, data) in enumerate(stats.items()):
        # Add mean line
        plt.hlines(y=data['mean'], xmin=i - 0.2, xmax=i + 0.2,
                   color='red', linestyle='--', linewidth=2)

        # Add text box with statistics
        stats_text = f"Mean: {data['mean']:.2f}\n"
        stats_text += f"Std: {data['std']:.2f}\n"
        stats_text += f"n: {data['n']}"

        plt.text(i, 7.2, stats_text,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                 ha='center', va='bottom',
                 fontweight='bold')

    # Add percentage of total responses
    total_responses = len(df)
    for i, (label, data) in enumerate(stats.items()):
        percentage = (data['n'] / total_responses) * 100
        plt.text(i, 0.5, f"{percentage:.1f}% of responses",
                 ha='center', va='bottom',
                 fontweight='bold', color='darkblue')

    # Customize the plot
    plt.title('CES Comparison: Clients With vs Without Ad Spend', pad=40)
    plt.xlabel('')
    plt.ylabel('CES Score')
    set_ces_y_axis(ax)

    # Add significance test result if applicable
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(with_ad_spend, without_ad_spend)
    significance_text = f"T-test p-value: {p_value:.4f}"
    if p_value < 0.05:
        significance_text += "*"
    if p_value < 0.01:
        significance_text += "*"
    if p_value < 0.001:
        significance_text += "*"

    plt.text(0.5, -0.5, significance_text,
             ha='center', va='top',
             transform=ax.transAxes,
             fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    plt.show()

def plot_ad_spend_analysis(df):
    """Plot key relationships between ad spend and CES."""
    # Only analyze records with ad spend
    ad_spend_data = df[df['AdSpendYN']].copy()

    if len(ad_spend_data) == 0:
        print("No ad spend data available for visualization")
        return

    plt.figure(figsize=(15, 5))

    # 1. CES by Ad Spend Quintile
    plt.subplot(1, 3, 1)
    if 'AdSpend_Quintile' in ad_spend_data.columns:
        sns.boxplot(x='AdSpend_Quintile', y='CES_Response_Value', data=ad_spend_data)
        plt.title('CES Distribution by Ad Spend Quintile')
        plt.xlabel('Ad Spend Quintile')
        plt.ylabel('CES Score')
        set_ces_y_axis(plt.gca())

    # 2. CES by Predefined Spend Brackets
    plt.subplot(1, 3, 2)
    if 'AdSpend_Bins' in ad_spend_data.columns:
        sns.boxplot(x='AdSpend_Bins', y='CES_Response_Value', data=ad_spend_data)
        plt.title('CES by Ad Spend Bracket')
        plt.xlabel('Monthly Ad Spend')
        plt.ylabel('CES Score')
        plt.xticks(rotation=45)
        set_ces_y_axis(plt.gca())

    # 3. Efficiency Categories
    plt.subplot(1, 3, 3)
    if 'Efficiency_Category' in ad_spend_data.columns:
        sns.boxplot(x='Efficiency_Category', y='CES_Response_Value', data=ad_spend_data)
        plt.title('CES by Efficiency Category')
        plt.xlabel('Efficiency Category')
        plt.ylabel('CES Score')
        plt.xticks(rotation=45)
        set_ces_y_axis(plt.gca())

    plt.tight_layout()
    plt.show()

def generate_all_visualization_calls(df, model, config, combinatorial_results):
    print("Generating visualizations...")

    viz_functions = [
        plot_ces_scores_by_response_group,
        lambda df: plot_boxplot_by_category(df, 'Response_Group', 'CES_Response_Value'),
        lambda df: plot_ces_distribution_by_group(df, 'Response_Group', 'CES_Response'),
        lambda df: plot_boxplot_by_category(df, 'ClientUser_Type', 'CES_Response_Value'),
        lambda df: plot_boxplot_by_category(df, 'Has_Partner', 'CES_Response_Value'),
        lambda df: plot_ces_distribution(df, 'CES_Response_Value'),
        lambda df: plot_feature_importances(model, df[config['modeling']['feature_columns']]),
        lambda df: plot_correlation_with_ces(df, 'CES_Response_Value'),
        lambda df: plot_mean_ces_by_db(df, 'db_number', 'CES_Response_Value'),
        lambda df: plot_boxplot_by_category(df, 'ClientUser_Type', 'CES_Response_Value'),
        lambda df: visualize_combinatorial_results(combinatorial_results),
        plot_combined_ces_distribution,
        lambda df: plot_ces_by_variable_and_response_group(df, 'ClientUser_Type'),
        plot_filtered_ces_distribution,
        plot_ad_spend_analysis,
        lambda df: generate_ad_spend_visualizations(df),
        plot_detailed_db_analysis,
        plot_database_performance_comparison
    ]

    for viz_func in viz_functions:
        try:
            viz_func(df)
            time.sleep(5)
            plt.close('all')
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            continue

    print("All visualizations generated successfully.")

# Updated `generate_all_visualizations` function
# def generate_all_visualizations(df, model, config, combinatorial_results):
#     print("Generating visualizations...")
#
#     # 0. CES Scores by Cohort
#     plot_ces_scores_by_response_group(df)
#     plot_boxplot_by_category(df, 'Response_Group', 'CES_Response_Value')
#
#     # 1. CES scores by decile across metrics (e.g., account age, leads per seat)
#     #plot_ces_by_ntile(df, 'account_age',10)
#     #plot_ces_by_ntile(df, 'leads_per_seat',4)
#     #plot_ces_by_ntile(df, 'account_age',10)
#     #plot_ces_by_ntile(df, 'leads_per_seat',4)
#
#     # 2. Percent distribution across CES responses by cohort
#     plot_ces_distribution_by_group(df, 'Response_Group', response_column='CES_Response')
#
#     # 3. CES trend over time
#     #plot_ces_trend_over_time(df, 'Response_Timestamp', 'CES_Response_Value')
#
#     # 4. CES by category (e.g., user type, partner status)
#     plot_boxplot_by_category(df, 'ClientUser_Type', 'CES_Response_Value')
#     plot_boxplot_by_category(df, 'Has_Partner', 'CES_Response_Value')
#
#     # 5. Distribution of CES responses
#     plot_ces_distribution(df, 'CES_Response_Value')
#
#     # 6. CES by time of day
#     #plot_ces_by_time_of_day(df, 'Response_Timestamp', 'CES_Response_Value')
#
#     # 7. CES by day of the week
#     #plot_ces_by_day_of_week(df, 'Response_Timestamp', 'CES_Response_Value')
#
#     # 8. Feature importances in the random forest model + SHAP summary plot
#     plot_feature_importances(model, df[config['modeling']['feature_columns']])
#
#     # 9. Correlation of numerical features with CES
#     plot_correlation_with_ces(df, target_column='CES_Response_Value')
#
#     # 10. Mean CES and response count by DB number
#     plot_mean_ces_by_db(df, 'db_number', 'CES_Response_Value')
#
#     # 11. Boxplot by DB and user type
#     #plot_boxplot_by_category(df, 'db_number', 'CES_Response_Value')
#     plot_boxplot_by_category(df, 'ClientUser_Type', 'CES_Response_Value')
#
#     # 12. Elasticity analysis of cohorts
#     #plot_elasticity_by_cohort(df, 'Response_Group', 'CES_Response_Value')
#
#     # 13. Combinatorial analysis of cohorts
#     visualize_combinatorial_results(combinatorial_results)
#
#     # 14. Identify top stable and elastic groups across all combinations
#     most_stable, most_elastic = identify_top_stable_elastic_groups(combinatorial_results)
#
#     ## 14a. Print most stable groups
#     print("\nMost Stable Groups across all combinations:")
#     print(most_stable.to_string(index=False))
#
#     ## 14b. Print most elastic groups
#     print("\nMost Elastic Groups across all combinations:")
#     print(most_elastic.to_string(index=False))
#
#     #15 Radar chart
#     #categories = ['ClientUser_Type', 'Response_Group']
#     #plot_ces_radar_chart(df, categories)
#
#     #16 Distribution Curves of user type per response group
#     plot_combined_ces_distribution(df)
#     #analyze_ces_distributions(df)
#
#     #17 USer Type Line Chart
#     plot_ces_by_variable_and_response_group(df, 'ClientUser_Type')
#     #version with new variable and title for reference
#     #plot_ces_by_variable_and_response_group(df, 'SomeVariable', 'Custom Chart Title')
#
#     #18 PM Distrubtion
#     plot_filtered_ces_distribution(df, 'CES_Response_Value')
#
#     #19 ad spend visualization
#     print("Generating ad spend analysis visualizations...")
#     print("haven't done this yet")
#     plot_ad_spend_analysis(df)
#     generate_ad_spend_visualizations(df)
#
#     #20 database comparisons
#     plot_database_performance_comparison(df)
#     plot_detailed_db_analysis(df)
#
#     #00 Sankey
#     #plot_ces_sankey_diagram(df, 'ClientUser_Type', 'CES_Response')
