import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_ind
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from utils import OutputFormatter


def analyze_ces_vs_churn(ces_data, churn_data_path, churn_window_days=1000):
    # Load churn data
    churn_data = pd.read_csv(churn_data_path)

    # Convert ka and SiteID in CAA to string and ensure consistent formatting
    ces_data['ka'] = ces_data['ka'].astype(str).str.strip()
    churn_data['SiteID in CAA'] = churn_data['SiteID in CAA'].astype(str).str.strip()

    #debug
    print(churn_data.head().to_dict())
    print(churn_data['SiteID in CAA'].dtype)



    # Ensure Response_Timestamp is in datetime format
    ces_data['Response_Timestamp'] = pd.to_datetime(ces_data['Response_Timestamp'])

    # Convert Cancellation Scheduled Date to datetime
    churn_data['Cancellation Scheduled Date'] = pd.to_datetime(churn_data['Cancellation Scheduled Date'])

    # Filter out churn data before February 1, 2024
    cutoff_date = pd.Timestamp('2024-02-01')
    churn_data = churn_data[churn_data['Cancellation Scheduled Date'] >= cutoff_date]

    # print(f"\nCES data shape after filtering: {ces_data.shape}")
    # print(f"Churn data shape after filtering: {churn_data.shape}")
    #
    # print(f"\nUnique ka in CES data: {ces_data['ka'].nunique()}")
    # print(f"Unique SiteIDs in churn data: {churn_data['SiteID in CAA'].nunique()}")
    #
    # print(f"\nCES data date range: {ces_data['Response_Timestamp'].min()} to {ces_data['Response_Timestamp'].max()}")
    # print(
    #     f"Churn data date range: {churn_data['Cancellation Scheduled Date'].min()} to {churn_data['Cancellation Scheduled Date'].max()}")

    print(OutputFormatter.format_data_overview(ces_data, churn_data))

    # Merge CES data with churn data based on SiteID in CAA and ka
    merged_data = pd.merge(ces_data, churn_data, left_on='ka', right_on='SiteID in CAA', how='left')

    print(f"\nMerged data shape: {merged_data.shape}")
    print(f"Null values in Cancellation Scheduled Date: {merged_data['Cancellation Scheduled Date'].isnull().sum()}")

    # Calculate days to churn
    merged_data['Days_to_Churn'] = (
                merged_data['Cancellation Scheduled Date'] - merged_data['Response_Timestamp']).dt.days

    # Count raw matches
    raw_matches = merged_data['SiteID in CAA'].notna().sum()
    print(f"\nRaw matches (ignoring dates): {raw_matches}")

    # Categorize matches
    future_churns = merged_data[(merged_data['SiteID in CAA'].notna()) & (merged_data['Days_to_Churn'] > 0)].shape[0]
    past_churns = merged_data[(merged_data['SiteID in CAA'].notna()) & (merged_data['Days_to_Churn'] <= 0)].shape[0]
    no_churn_date = \
    merged_data[(merged_data['SiteID in CAA'].notna()) & (merged_data['Cancellation Scheduled Date'].isna())].shape[0]

    print(f"\nMatches breakdown:")
    print(f"Future churns: {future_churns}")
    print(f"Past churns: {past_churns}")
    print(f"Matches without churn date: {no_churn_date}")

    # Check for duplicate matches
    duplicate_matches = merged_data[merged_data.duplicated(['ka', 'SiteID in CAA'], keep=False)]
    print(f"\nNumber of duplicate matches: {duplicate_matches.shape[0]}")
    if duplicate_matches.shape[0] > 0:
        print("\nSample of duplicate matches:")
        print(duplicate_matches[['ka', 'SiteID in CAA', 'Response_Timestamp', 'Cancellation Scheduled Date']].head())

    # Modified churn detection logic
    merged_data['Churned'] = merged_data['SiteID in CAA'].notna() & (
            (merged_data['Cancellation Scheduled Date'].isna()) |  # Include matches without a cancellation date
            (merged_data['Cancellation Scheduled Date'] >= merged_data['Response_Timestamp'])  # Future churns
    )

    churned_count = merged_data['Churned'].sum()
    print(f"\nTotal matches considered as churned: {churned_count}")

    # Print sample of matched data
    # print("\nSample of matched data:")
    # print(merged_data[merged_data['SiteID in CAA'].notna()][
    #           ['ka', 'SiteID in CAA', 'Response_Timestamp', 'Cancellation Scheduled Date', 'Days_to_Churn',
    #            'Churned']].head(20))

    # Distribution of days to churn for churned clients
    churned_clients = merged_data[merged_data['Churned'] == True]
    print("\nDays to Churn statistics for churned clients:")
    print(churned_clients['Days_to_Churn'].describe())

    # Calculate CES Statistics
    avg_ces_churned = merged_data[merged_data['Churned'] == True]['CES_Response_Value'].mean()
    avg_ces_non_churned = merged_data[merged_data['Churned'] == False]['CES_Response_Value'].mean()

    # Calculate Churn Rates by Time Window
    churn_rates = {}
    for window in [30, 90, 180, 365]:
        rate = (merged_data['Days_to_Churn'].between(0, window)).mean() * 100
        churn_rates[f"Within {window} days"] = rate

    # Calculate CES Score Range Rates
    ces_range_rates = {}
    for score_range in [(1, 3), (4, 5), (6, 7)]:
        denominator = merged_data[merged_data['CES_Response_Value'].between(score_range[0], score_range[1])].shape[0]
        if denominator > 0:  # Avoid division by zero
            numerator = merged_data[
                (merged_data['CES_Response_Value'].between(score_range[0], score_range[1])) &
                (merged_data['Churned'] == True)
                ].shape[0]
            rate = (numerator / denominator) * 100
            ces_range_rates[f"CES {score_range[0]}-{score_range[1]}"] = rate

    # Print formatted analysis results
    print("\n" + "=" * 50)
    print(OutputFormatter.format_churn_analysis(
        avg_ces_churned,
        avg_ces_non_churned,
        churn_rates
    ))

    # Add CES Score Range Analysis
    print("\nChurn Rate by CES Score Range")
    print("-" * 40)
    print(f"{'Score Range':<20} {'Rate':>10}")
    print("-" * 40)
    for range_label, rate in ces_range_rates.items():
        print(f"{range_label:<20} {rate:>9.1f}%")

    # Print correlation information in a clear format
    correlation = churned_clients['CES_Response_Value'].corr(churned_clients['Days_to_Churn'])
    print("\nCorrelation Analysis")
    print("-" * 40)
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 40)
    print(f"{'CES vs Days to Churn':<30} {correlation:>10.3f}")

    return merged_data


def estimate_propensity_score(df, target_col, covariates):
    """
    Estimate propensity scores using logistic regression, handling missing values.

    Parameters:
    df (pandas.DataFrame): The dataset with the target variable (churn) and covariates
    target_col (str): The name of the target column (Churned)
    covariates (list): List of covariate column names to include in the model

    Returns:
    pandas.Series: Propensity scores
    logit_summary: The summary of the logistic regression model
    """
    # One-hot encode categorical variables
    X = pd.get_dummies(df[covariates], drop_first=True)
    X = sm.add_constant(X)  # Add a constant term to the model

    # Handle missing or infinite values in covariates
    if X.isnull().any().any():
        print("Missing values detected in covariates, filling with mean values.")
        X = X.fillna(X.mean())  # You can also use other imputation strategies if necessary

    # Define the target variable
    y = df[target_col].astype(int)

    # Fit logistic regression model
    logit_model = sm.Logit(y, X)

    # Try fitting the model, and catch any errors related to missing or inf values
    try:
        logit_result = logit_model.fit()
    except Exception as e:
        print(f"Error during logistic regression: {e}")
        raise

    # Calculate propensity scores
    propensity_scores = logit_result.predict(X)

    return propensity_scores, logit_result.summary()


def nearest_neighbor_matching(df, target_col='Churned', score_col='propensity_score', k=1):
    """
    Perform nearest-neighbor matching based on propensity scores.

    Parameters:
    df (pandas.DataFrame): The dataset with propensity scores and churn information
    target_col (str): The name of the target column indicating churn (True/False)
    score_col (str): The name of the column with propensity scores
    k (int): Number of neighbors to match (default is 1 for nearest neighbor)

    Returns:
    pandas.DataFrame: Dataframe with matched pairs
    """
    # Separate churned and non-churned groups
    churned = df[df[target_col] == True].copy()
    non_churned = df[df[target_col] == False].copy()

    # Fit nearest-neighbor model using propensity scores of non-churned
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(non_churned[[score_col]])

    # Find nearest neighbors for churned clients
    distances, indices = nn.kneighbors(churned[[score_col]])

    # Map the indices of nearest neighbors to get matched data
    matched_non_churned = non_churned.iloc[indices.flatten()]

    # Combine the matched churned and non-churned data
    matched_churned = churned.reset_index(drop=True)
    matched_non_churned = matched_non_churned.reset_index(drop=True)

    # Return matched dataframe
    matched_data = pd.concat([matched_churned, matched_non_churned], axis=0).reset_index(drop=True)
    return matched_data

def perform_psm(df, propensity_scores, churn_col, n_neighbors=1):
    """
    Perform Propensity Score Matching using nearest neighbor matching.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    propensity_scores (pd.Series): The estimated propensity scores.
    churn_col (str): The column indicating whether the client churned.
    n_neighbors (int): Number of nearest neighbors to match.

    Returns:
    pd.DataFrame: DataFrame with matched pairs of churned and non-churned clients.
    """
    # Separate the data into churned and non-churned
    churned = df[df[churn_col] == 1].copy()
    non_churned = df[df[churn_col] == 0].copy()

    # Nearest Neighbor Matching
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(non_churned[['propensity_score']])

    # Find the nearest neighbors for each churned client
    distances, indices = nn.kneighbors(churned[['propensity_score']])

    # Create a matched dataset
    matched_non_churned = non_churned.iloc[indices.flatten()]

    # Combine churned and matched non-churned clients
    matched_df = pd.concat([churned, matched_non_churned])

    return matched_df


# Main function to run PSM
def run_psm_analysis(df, covariates, k=1):
    """
    Perform propensity score matching analysis.

    Parameters:
    df (pandas.DataFrame): The dataset with CES and churn data
    covariates (list): List of covariates to match on
    k (int): Number of neighbors for nearest-neighbor matching (default is 1)

    Returns:
    matched_data (pandas.DataFrame): Data with matched pairs
    logit_summary: Summary of the logistic regression model
    """
    # Step 1: Estimate the propensity scores
    propensity_scores, logit_summary = estimate_propensity_score(df, target_col='Churned', covariates=covariates)

    # Step 2: Add propensity scores to the dataset
    df['propensity_score'] = propensity_scores

    # Step 3: Perform nearest-neighbor matching based on the propensity score
    matched_data = nearest_neighbor_matching(df, target_col='Churned', score_col='propensity_score', k=k)

    return matched_data, logit_summary


def visualize_ces_vs_churn(matched_data):
    """
    Visualize the CES scores for churned and non-churned clients after matching.

    Parameters:
    matched_data (pandas.DataFrame): The dataset containing CES scores and churn labels after PSM
    """
    # Ensure that the matched_data has the necessary columns
    if 'CES_Response_Value' not in matched_data.columns or 'Churned' not in matched_data.columns:
        raise ValueError("The matched data must contain 'CES_Response_Value' and 'Churned' columns")

    # Plot the distribution of CES scores for churned and non-churned clients
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churned', y='CES_Response_Value', data=matched_data)
    plt.title('CES Scores by Churned/Non-Churned Clients')
    plt.xlabel('Churn Status (0 = Not Churned, 1 = Churned)')
    plt.ylabel('CES Response Value')
    plt.tight_layout()
    plt.show()

    # Optionally, you can also show the density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=matched_data[matched_data['Churned'] == 0]['CES_Response_Value'], label='Non-Churned', shade=True)
    sns.kdeplot(data=matched_data[matched_data['Churned'] == 1]['CES_Response_Value'], label='Churned', shade=True)
    plt.title('Density Plot of CES Scores for Churned vs Non-Churned Clients')
    plt.xlabel('CES Response Value')
    plt.xlim(1, 7)
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the mean CES score for churned and non-churned clients
    ces_mean = matched_data.groupby('Churned')['CES_Response_Value'].mean().reset_index()
    plt.figure(figsize=(10, 6))

    sns.barplot(x='Churned', y='CES_Response_Value', data=ces_mean)
    plt.title('Mean CES Score for Churned and Non-Churned Clients')
    plt.xlabel('Churn Status (0 = Not Churned, 1 = Churned)')
    plt.ylabel('Mean CES Response Value')
    plt.ylim(0, 7)

    plt.tight_layout()
    plt.show()


def count_ces_scores_from_churned_clients(ces_data, churn_data_path, churn_window_days=1000):
    # Load churn data
    churn_data = pd.read_csv(churn_data_path)

    # print(f"Original CES data shape: {ces_data.shape}")
    # print(f"Original Churn data shape: {churn_data.shape}")

    # Convert ka and SiteID in CAA to string and ensure consistent formatting
    ces_data['ka'] = ces_data['ka'].astype(str).str.strip()
    churn_data['SiteID in CAA'] = churn_data['SiteID in CAA'].astype(str).str.strip()

    # Ensure Response_Timestamp is in datetime format
    ces_data['Response_Timestamp'] = pd.to_datetime(ces_data['Response_Timestamp'])

    # Convert Cancellation Scheduled Date to datetime
    churn_data['Cancellation Scheduled Date'] = pd.to_datetime(churn_data['Cancellation Scheduled Date'])

    # Filter out churn data before February 1, 2024
    cutoff_date = pd.Timestamp('2024-02-01')
    churn_data = churn_data[churn_data['Cancellation Scheduled Date'] >= cutoff_date]

    print(f"\nCES data shape after filtering: {ces_data.shape}")
    print(f"Churn data shape after filtering: {churn_data.shape}")

    # Merge CES data with churn data based on SiteID in CAA and ka
    merged_data = pd.merge(ces_data, churn_data, left_on='ka', right_on='SiteID in CAA', how='left')

    print(f"\nMerged data shape: {merged_data.shape}")

    # Calculate days to churn
    merged_data['Days_to_Churn'] = (merged_data['Cancellation Scheduled Date'] - merged_data['Response_Timestamp']).dt.days

    # Modified churn detection logic
    merged_data['Churned'] = merged_data['SiteID in CAA'].notna() & (
            (merged_data['Cancellation Scheduled Date'].isna()) |  # Include matches without a cancellation date
            (merged_data['Cancellation Scheduled Date'] >= merged_data['Response_Timestamp'])  # Future churns
    )

    # Count CES scores from churned clients
    churned_ces_count = merged_data['Churned'].sum()

    print(f"\nNumber of CES scores from clients that eventually churned: {churned_ces_count}")

    # Additional statistics
    print(f"\nTotal number of CES scores: {merged_data.shape[0]}")
    print(f"Percentage of CES scores from churned clients: {(churned_ces_count / merged_data.shape[0]) * 100:.2f}%")

    # Distribution of CES scores for churned vs non-churned clients
    print("\nCES score statistics for churned clients:")
    print(merged_data[merged_data['Churned']]['CES_Response_Value'].describe())

    print("\nCES score statistics for non-churned clients:")
    print(merged_data[~merged_data['Churned']]['CES_Response_Value'].describe())

    return churned_ces_count, merged_data


def logistic_regression_ces_on_churn(ces_data, churn_data_path, covariates, churn_window_days=1000):
    # Load churn data
    churn_data = pd.read_csv(churn_data_path)

    print(f"Original CES data shape: {ces_data.shape}")
    print(f"Original Churn data shape: {churn_data.shape}")

    # Print column names for debugging
    # print("CES data columns:", ces_data.columns.tolist())
    # print("Churn data columns:", churn_data.columns.tolist())

    # Convert ka and SiteID in CAA to string and ensure consistent formatting
    ces_data['ka'] = ces_data['ka'].astype(str).str.strip()
    churn_data['SiteID in CAA'] = churn_data['SiteID in CAA'].astype(str).str.strip()

    # Ensure Response_Timestamp is in datetime format
    ces_data['Response_Timestamp'] = pd.to_datetime(ces_data['Response_Timestamp'])

    # Check if 'Cancellation Scheduled Date' exists in churn_data
    if 'Cancellation Scheduled Date' in churn_data.columns:
        # Convert Cancellation Scheduled Date to datetime
        churn_data['Cancellation Scheduled Date'] = pd.to_datetime(churn_data['Cancellation Scheduled Date'])

        # Filter out churn data before February 1, 2024
        cutoff_date = pd.Timestamp('2024-02-01')
        churn_data = churn_data[churn_data['Cancellation Scheduled Date'] >= cutoff_date]
    else:
        print("Warning: 'Cancellation Scheduled Date' not found in churn data.")

    # print(f"\nCES data shape after filtering: {ces_data.shape}")
    # print(f"Churn data shape after filtering: {churn_data.shape}")

    # Merge CES data with churn data based on SiteID in CAA and ka
    merged_data = pd.merge(ces_data, churn_data, left_on='ka', right_on='SiteID in CAA', how='left')

    # print(f"\nMerged data shape: {merged_data.shape}")
    # print("Merged data columns:", merged_data.columns.tolist())

    # Check if necessary columns exist before calculating Days_to_Churn
    if 'Cancellation Scheduled Date' in merged_data.columns and 'Response_Timestamp' in merged_data.columns:
        merged_data['Days_to_Churn'] = (merged_data['Cancellation Scheduled Date'] - merged_data['Response_Timestamp']).dt.days
    else:
        print("Warning: Unable to calculate 'Days_to_Churn'. Required columns not found.")

    # Modified churn detection logic
    if 'SiteID in CAA' in merged_data.columns and 'Cancellation Scheduled Date' in merged_data.columns:
        merged_data['Churned'] = merged_data['SiteID in CAA'].notna() & (
                (merged_data['Cancellation Scheduled Date'].isna()) |  # Include matches without a cancellation date
                (merged_data['Cancellation Scheduled Date'] >= merged_data['Response_Timestamp'])  # Future churns
        )
    elif 'Churned' not in merged_data.columns:
        print("Warning: Unable to determine churn status. Using a placeholder 'Churned' column.")
        merged_data['Churned'] = False  # Placeholder, adjust as needed

    # Ensure all covariates exist in the merged data
    existing_covariates = [cov for cov in covariates if cov in merged_data.columns]
    if len(existing_covariates) < len(covariates):
        print(f"Warning: Some covariates not found in data. Using only: {existing_covariates}")

    # Drop rows with missing values in covariates or CES_Response_Value
    merged_data = merged_data.dropna(subset=['CES_Response_Value'] + existing_covariates)

    # Convert categorical variables into dummy/indicator variables
    categorical_covariates = [cov for cov in existing_covariates if merged_data[cov].dtype == 'object']
    merged_data = pd.get_dummies(merged_data, columns=categorical_covariates, drop_first=True)

    # Update covariates list with dummy variable names
    dummy_covariates = [col for col in merged_data.columns if any(cov in col for cov in categorical_covariates)]
    updated_covariates = [cov for cov in existing_covariates if cov not in categorical_covariates] + dummy_covariates

    # Prepare features (X) and target variable (y)
    X = merged_data[['CES_Response_Value'] + updated_covariates]
    X = sm.add_constant(X)  # Add a constant term for the intercept
    y = merged_data['Churned'].astype(int)

    # Fit logistic regression model
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    # Display the summary
    print(result.summary())

    # Additional analysis (assuming sklearn is imported)
    print("\nOdds Ratios:")
    print(np.exp(result.params))

    print("\nConfidence Intervals:")
    print(result.conf_int())

    # Model evaluation
    y_pred = result.predict(X)
    y_pred_binary = (y_pred > 0.5).astype(int)

    print("\nModel Accuracy:", accuracy_score(y, y_pred_binary))
    print("\nClassification Report:")
    print(classification_report(y, y_pred_binary))

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return result, merged_data