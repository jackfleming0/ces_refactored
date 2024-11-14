import logging
from data_loading import load_config, load_data
from data_cleaning import clean_data
from feature_engineering import create_features
from eda import aggregate_ces_by_column, plot_ces_distribution, analyze_ces_vs_content_pages, analyze_ces_vs_ad_spend, format_ad_analysis_results, format_category_stats

from churn_analysis import (analyze_ces_vs_churn,
                            run_psm_analysis,
                            visualize_ces_vs_churn,
                            logistic_regression_ces_on_churn)
from statistical_tests import t_test_between_groups, anova_test, tukey_hsd_test
from modeling import prepare_data_for_modeling, train_random_forest, evaluate_model, cross_validate_model
from utils import bin_numerical_column
from sklearn.model_selection import train_test_split
from visualization import generate_all_visualizations
from stability_analysis import perform_combinatorial_analysis
import pandas as pd




def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load configurations
    config = load_config()

    # Load in and clean data. Make sure to update the "data/ces_data.csv" file each usage with the most up to date.
    ces_data = load_data(config['data']['ces_data_path'])
    ces_data_clean = clean_data(ces_data, config['preprocessing']['date_columns'],
                                config['preprocessing']['date_formats'])

    # Feature engineering - creating calculated variables
    ces_data_fe = create_features(
        ces_data_clean,
        config['cohorts'],
        config=config,  # Added config parameter
        ad_spend_path='../data/All Accounts-Table 1.csv'  # Added ad spend path
    )

    #debugging lines
    print(ces_data.head().to_dict())
    print(ces_data['FE_Site_ID'].dtype)

    # Binning numerical columns - configured in config.yaml
    for column, n_bins in config['columns_to_bin'].items():
        ces_data_fe = bin_numerical_column(ces_data_fe, column, n_bins)

    # EDA
    print("\nPerforming exploratory data analysis...")

    # Basic CES analysis
    aggregate_ces = aggregate_ces_by_column(ces_data_fe, 'ClientUser_Type', 'User Type')
    plot_ces_distribution(ces_data_fe, 'ClientUser_Type', save_path='../outputs/figures/ces_distribution.png')

    # Content pages analysis
    print("\nAnalyzing CES vs content pages...")
    ces_vs_content_pages_analysis = analyze_ces_vs_content_pages(ces_data_fe)

    # Ad spend analysis
    # Ad spend analysis
    print("\nPerforming ad spend analysis...")
    ad_spend_data, correlations, stats_summary = analyze_ces_vs_ad_spend(ces_data_fe)

    # Format and display results
    format_ad_analysis_results(correlations, stats_summary)

    # Save ad spend correlations
    if correlations:
        correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with CES'])
        correlation_df.to_csv('../outputs/analyses/ad_spend_correlations.csv')
        print(f"\nAd spend correlations saved to '../outputs/analyses/ad_spend_correlations.csv'")

        # Log top correlations
        print("\nTop 3 strongest ad spend correlations:")
        for metric, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
            print(f"{metric}: {corr:.3f}")

    # Log key statistical findings
    if stats_summary:
        print("\nStatistical test results:")
        for test, results in stats_summary.items():
            if isinstance(results, dict):
                if 'f_stat' in results and 'p_value' in results:
                    print(f"{test}: F={results['f_stat']:.2f}, p={results['p_value']:.4f}")
                else:
                    print(f"{test}: {results}")
            else:
                print(f"{test}: {results}")

    # Statistical tests
    t_stat, p_value = t_test_between_groups(ces_data_fe, 'Response_Group', 'CES_Response_Value', 'Group 1', 'Group 2')
    print(t_stat)
    print(p_value)
    t_stat, p_value = t_test_between_groups(ces_data_fe, 'Response_Group', 'CES_Response_Value', 'Group 3', 'Group 4')
    print(t_stat)
    print(p_value)
    t_stat, p_value = t_test_between_groups(ces_data_fe, 'Response_Group', 'CES_Response_Value', 'Group 2', 'Group 4')
    print(t_stat)
    print(p_value)
    t_stat, p_value = t_test_between_groups(ces_data_fe, 'Response_Group', 'CES_Response_Value', 'Group 1', 'Group 4')
    print(t_stat)
    print(p_value)

    f_stat, p_value = anova_test(ces_data_fe, 'Response_Group', 'CES_Response_Value')
    print(f_stat)
    print(p_value)

    tukey_results = tukey_hsd_test(ces_data_fe, 'Response_Group', 'CES_Response_Value')
    print(tukey_results)

    f_stat, p_value = anova_test(ces_data_fe, 'db_number', 'CES_Response_Value')
    print(f_stat)
    print(p_value)

    tukey_results = tukey_hsd_test(ces_data_fe, 'db_number', 'CES_Response_Value')
    print(tukey_results)

    f_stat, p_value = anova_test(ces_data_fe, 'db_cohort', 'CES_Response_Value')
    print(f_stat)
    print(p_value)

    tukey_results = tukey_hsd_test(ces_data_fe, 'db_cohort', 'CES_Response_Value')
    print(tukey_results)

    f_stat, p_value = anova_test(ces_data_fe, 'ClientUser_Type', 'CES_Response_Value')
    print(f_stat)
    print(p_value)

    tukey_results = tukey_hsd_test(ces_data_fe, 'ClientUser_Type', 'CES_Response_Value')
    print(tukey_results)

    # Churn Analysis
    # Step 1: Merge CES data with churn data
    ces_vs_churn_analysis = analyze_ces_vs_churn(ces_data_fe, '../data/churned_clients.csv')

    # Step 2: Define covariates for PSM analysis
    covariates = ['account_age', 'leads_per_seat','ClientUser_Cnt','Client_#Leads','Client_#Logins','30DayPageViews','db_number','Plan_Amt_Dec23']

    # Step 3: Run Propensity Score Matching using the merged data (ces_vs_churn_analysis)
    matched_data, logit_summary = run_psm_analysis(ces_vs_churn_analysis, covariates=covariates)

    # Step 4: Visualize and interpret results (if needed)
    visualize_ces_vs_churn(matched_data)

    # Step 5: Print scores
    print(logit_summary)  # View the logistic regression summary
    print(matched_data.head())  # View the logistic regression summary

    # Step 6: Logistic Regression
    logit_result = logistic_regression_ces_on_churn(matched_data, '../data/churned_clients.csv', covariates)
    print(logit_result)

    # Modeling
    feature_columns = ['leads_per_seat', 'account_age', 'ClientUser_Cnt', 'Client_#Logins', '30DayPageViews',
                       'Plan_Amt_Dec23']
    target_column = 'CES_Response_Value'
    X, y = prepare_data_for_modeling(ces_data_fe, feature_columns, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    rmse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Test RMSE: {rmse}, R2: {r2}")

    # Stability analysis
    ## Extract analysis parameters from config file
    base_columns = config['analysis']['base_columns']
    additional_columns = config['analysis']['additional_columns']
    time_column = config['analysis']['time_column']
    ces_column = config['analysis']['ces_column']
    min_responses = config['analysis']['min_responses']
    min_combo = config['analysis']['min_combo']
    max_combo = config['analysis']['max_combo']
    ## Run Analysis
    combinatorial_results = perform_combinatorial_analysis(
        ces_data_fe, base_columns, additional_columns, time_column, ces_column,
        min_responses, min_combo, max_combo
    )


    # Cross-validation
    rmse_scores = cross_validate_model(model, X, y, cv=5)
    print(f"Cross-validation RMSE scores: {rmse_scores}")

    # Generate all visualizations
    generate_all_visualizations(ces_data_fe, model, config, combinatorial_results)


if __name__ == "__main__":
    main()