import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm

def analyze_ces_vs_churn(ces_data, churn_data_path):
    # Load churn data
    churn_data = pd.read_csv(churn_data_path)

    # Convert FE_Site_ID and SiteID in CAA to string for consistent merging
    ces_data['FE_Site_ID'] = ces_data['FE_Site_ID'].astype(str)
    churn_data['SiteID in CAA'] = churn_data['SiteID in CAA'].astype(str)

    # Merge CES data with churn data based on SiteID in CAA and FE_Site_ID
    merged_data = pd.merge(ces_data, churn_data, left_on='FE_Site_ID', right_on='SiteID in CAA', how='left')

    # Create a new column to identify churned vs non-churned clients
    merged_data['Churned'] = merged_data['SiteID in CAA'].notna()

    # Plot CES Response distribution for churned vs non-churned clients
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churned', y='CES_Response_Value', data=merged_data)
    plt.title('CES Responses for Churned vs Non-Churned Clients')
    plt.xlabel('Churned')
    plt.ylabel('CES Response Value')
    plt.tight_layout()
    plt.show()

    # Optional: Perform a statistical test (e.g., t-test) to compare CES scores between churned and non-churned clients
    churned_ces = merged_data[merged_data['Churned'] == True]['CES_Response_Value']
    non_churned_ces = merged_data[merged_data['Churned'] == False]['CES_Response_Value']

    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(churned_ces.dropna(), non_churned_ces.dropna(), equal_var=False)
    print(f"T-test between churned and non-churned clients: t={t_stat}, p={p_value}")

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
    plt.tight_layout()
    plt.show()


def logistic_regression_ces_on_churn(ces_data, churn_data_path, covariates):
    # Step 1: Merge CES data with churn data
    churn_data = pd.read_csv(churn_data_path)
    merged_data = pd.merge(ces_data, churn_data, left_on='FE_Site_ID', right_on='SiteID in CAA', how='left')

    # Step 2: Drop rows with missing values (if any)
    merged_data = merged_data.dropna(subset=['CES_Response_Value', 'Churned'] + covariates)

    # Step 3: Convert categorical variables (like 'ClientUser_Type') into dummy/indicator variables
    merged_data = pd.get_dummies(merged_data, columns=['ClientUser_Type'], drop_first=True)

    # Step 4: Define X (covariates + CES score) and y (Churned)
    X = merged_data[['CES_Response_Value'] + covariates]
    X = sm.add_constant(X)  # Add a constant term for the intercept
    y = merged_data['Churned'].astype(int)  # Ensure y is numeric (0/1 for logistic regression)

    # Step 5: Fit logistic regression model
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    # Step 6: Display the summary
    print(result.summary())

    return result