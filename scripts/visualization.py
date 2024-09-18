import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import math
from stability_analysis import identify_top_stable_elastic_groups


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

def plot_ces_by_decile(df, column, target='CES_Response_Value'):
    df[f'{column}_decile'] = pd.qcut(df[column], 10)
    mean_ces_by_decile = df.groupby(f'{column}_decile')[target].mean()
    overall_mean = df[target].mean()

    plt.figure(figsize=(10, 6))
    ax = mean_ces_by_decile.plot(kind='bar')

    # Adding value labels to each bar
    for i, v in enumerate(mean_ces_by_decile):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center')

    # Adding horizontal line for overall mean
    plt.axhline(overall_mean, color='red', linestyle='--', label=f'Mean CES: {overall_mean:.2f}')
    plt.title(f'CES by {column.capitalize()} Decile')
    plt.xlabel(f'{column.capitalize()} Decile')
    plt.ylabel('Mean CES Score')
    plt.xticks(rotation=45)
    set_ces_y_axis(ax)  # Set y-axis to always run 0-7
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ces_distribution_by_group(df, group_column, response_column='CES_Response'):
    grouped_data = pd.crosstab(df[group_column], df[response_column], normalize='index') * 100
    ax = grouped_data.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10, 6))

    plt.ylabel('Percentage')
    plt.title(f'Percentage Distribution of {response_column} by {group_column}')
    plt.xticks(rotation=0)
    plt.legend(title=response_column)
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

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(df_mean.index, df_count, color='lightblue', label='Count')
    ax2 = ax1.twinx()
    ax2.plot(df_mean.index, df_mean, color='orange', label='Mean CES', marker='o')

    ax1.set_xlabel('DB Number')
    ax1.set_ylabel('Count')
    ax2.set_ylabel('Mean CES')
    set_ces_y_axis(ax2)  # Set y-axis to always run 0-7

    plt.title('Mean CES and Response Count by DB Number')
    plt.tight_layout()
    plt.show()


def plot_boxplot_by_category(df, category_column, target_column='CES_Response_Value'):
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=category_column, y=target_column, data=df)
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


# Updated `generate_all_visualizations` function
def generate_all_visualizations(df, model, config, combinatorial_results):
    print("Generating visualizations...")

    # 0. CES Scores by Cohort
    plot_ces_scores_by_response_group(df)
    plot_boxplot_by_category(df, 'Response_Group', 'CES_Response_Value')

    # 1. CES scores by decile across metrics (e.g., account age, leads per seat)
    plot_ces_by_decile(df, 'account_age')
    plot_ces_by_decile(df, 'leads_per_seat')

    # 2. Percent distribution across CES responses by cohort
    plot_ces_distribution_by_group(df, 'Response_Group', response_column='CES_Response')

    # 3. CES trend over time
    plot_ces_trend_over_time(df, 'Response_Timestamp', 'CES_Response_Value')

    # 4. CES by category (e.g., user type, partner status)
    plot_boxplot_by_category(df, 'ClientUser_Type', 'CES_Response_Value')
    plot_boxplot_by_category(df, 'Has_Partner', 'CES_Response_Value')

    # 5. Distribution of CES responses
    plot_ces_distribution(df, 'CES_Response_Value')

    # 6. CES by time of day
    plot_ces_by_time_of_day(df, 'Response_Timestamp', 'CES_Response_Value')

    # 7. CES by day of the week
    plot_ces_by_day_of_week(df, 'Response_Timestamp', 'CES_Response_Value')

    # 8. Feature importances in the random forest model + SHAP summary plot
    plot_feature_importances(model, df[config['modeling']['feature_columns']])

    # 9. Correlation of numerical features with CES
    plot_correlation_with_ces(df, target_column='CES_Response_Value')

    # 10. Mean CES and response count by DB number
    plot_mean_ces_by_db(df, 'db_number', 'CES_Response_Value')

    # 11. Boxplot by DB and user type
    plot_boxplot_by_category(df, 'db_number', 'CES_Response_Value')
    plot_boxplot_by_category(df, 'ClientUser_Type', 'CES_Response_Value')

    # 12. Elasticity analysis of cohorts
    plot_elasticity_by_cohort(df, 'Response_Group', 'CES_Response_Value')

    # 13. Combinatorial analysis of cohorts
    visualize_combinatorial_results(combinatorial_results)

    # 14. Identify top stable and elastic groups across all combinations
    most_stable, most_elastic = identify_top_stable_elastic_groups(combinatorial_results)

    ## 14a. Print most stable groups
    print("\nMost Stable Groups across all combinations:")
    print(most_stable.to_string(index=False))

    ## 14b. Print most elastic groups
    print("\nMost Elastic Groups across all combinations:")
    print(most_elastic.to_string(index=False))


    print("All visualizations generated successfully.")