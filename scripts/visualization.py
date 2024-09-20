import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import math
from stability_analysis import identify_top_stable_elastic_groups
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go




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

    #This produces a line chart.


    # Ensure 'Response_Group' is in the correct order
    group_order = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

    # Calculate mean CES for each combination of the variable and Response_Group
    grouped_data = df.groupby([variable_column, 'Response_Group'])['CES_Response_Value'].mean().unstack()

    # Reorder columns based on the specified order
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

# Updated `generate_all_visualizations` function
def generate_all_visualizations(df, model, config, combinatorial_results):
    print("Generating visualizations...")

    # 0. CES Scores by Cohort
    plot_ces_scores_by_response_group(df)
    plot_boxplot_by_category(df, 'Response_Group', 'CES_Response_Value')

    # 1. CES scores by decile across metrics (e.g., account age, leads per seat)
    #plot_ces_by_ntile(df, 'account_age',10)
    #plot_ces_by_ntile(df, 'leads_per_seat',4)
    #plot_ces_by_ntile(df, 'account_age',10)
    #plot_ces_by_ntile(df, 'leads_per_seat',4)

    # 2. Percent distribution across CES responses by cohort
    plot_ces_distribution_by_group(df, 'Response_Group', response_column='CES_Response')

    # 3. CES trend over time
    #plot_ces_trend_over_time(df, 'Response_Timestamp', 'CES_Response_Value')

    # 4. CES by category (e.g., user type, partner status)
    plot_boxplot_by_category(df, 'ClientUser_Type', 'CES_Response_Value')
    plot_boxplot_by_category(df, 'Has_Partner', 'CES_Response_Value')

    # 5. Distribution of CES responses
    plot_ces_distribution(df, 'CES_Response_Value')

    # 6. CES by time of day
    #plot_ces_by_time_of_day(df, 'Response_Timestamp', 'CES_Response_Value')

    # 7. CES by day of the week
    #plot_ces_by_day_of_week(df, 'Response_Timestamp', 'CES_Response_Value')

    # 8. Feature importances in the random forest model + SHAP summary plot
    plot_feature_importances(model, df[config['modeling']['feature_columns']])

    # 9. Correlation of numerical features with CES
    plot_correlation_with_ces(df, target_column='CES_Response_Value')

    # 10. Mean CES and response count by DB number
    plot_mean_ces_by_db(df, 'db_number', 'CES_Response_Value')

    # 11. Boxplot by DB and user type
    #plot_boxplot_by_category(df, 'db_number', 'CES_Response_Value')
    plot_boxplot_by_category(df, 'ClientUser_Type', 'CES_Response_Value')

    # 12. Elasticity analysis of cohorts
    #plot_elasticity_by_cohort(df, 'Response_Group', 'CES_Response_Value')

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

    #15 Radar chart
    #categories = ['ClientUser_Type', 'Response_Group']
    #plot_ces_radar_chart(df, categories)

    #16 Distribution Curves of user type per response group
    plot_combined_ces_distribution(df)
    #analyze_ces_distributions(df)

    #17 USer Type Line Chart
    plot_ces_by_variable_and_response_group(df, 'ClientUser_Type')
    #version with new variable and title for reference
    #plot_ces_by_variable_and_response_group(df, 'SomeVariable', 'Custom Chart Title')

    #18 PM Distrubtion
    plot_filtered_ces_distribution(df, 'CES_Response_Value')

    #00 Sankey
    #plot_ces_sankey_diagram(df, 'ClientUser_Type', 'CES_Response')

    print("All visualizations generated successfully.")