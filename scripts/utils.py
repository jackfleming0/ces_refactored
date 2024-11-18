import pandas as pd
import logging
from dateutil.parser import parse
import numpy as np

def parse_dates(date_str, date_formats):
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    logging.warning(f"Unable to parse date: {date_str}")
    return pd.NaT

def bin_numerical_column(df, column, n_bins=10, labels=None, binning_strategy='equal'):
    if labels is None:
        labels = [f'Bin {i+1}' for i in range(n_bins)]
    try:
        if binning_strategy == 'equal':
            df[f'{column}_binned'] = pd.cut(df[column], bins=n_bins, labels=labels, include_lowest=True)
        elif binning_strategy == 'quantile':
            df[f'{column}_binned'] = pd.qcut(df[column], q=n_bins, labels=labels, duplicates='drop')
        else:
            raise ValueError("Invalid binning_strategy. Choose 'equal' or 'quantile'.")
    except Exception as e:
        logging.error(f"Error binning column {column}: {e}")
    return df


class OutputFormatter:
    @staticmethod
    def format_correlations(correlations_dict, title="Correlation Analysis"):
        """
        Format correlation results into a readable table.

        Args:
            correlations_dict (dict): Dictionary of metric names and their correlation values
            title (str): Title for the correlation table

        Returns:
            str: Formatted correlation table as string
        """
        # Sort correlations by absolute value
        sorted_correlations = sorted(
            correlations_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Calculate dynamic spacing
        max_metric_length = max(len(str(metric)) for metric, _ in sorted_correlations)
        metric_width = max(max_metric_length, 20)

        # Create the formatted string
        lines = [
            title,
            "-" * (metric_width + 15),
            f"{'Metric':<{metric_width}} {'Correlation':>10}",
            "-" * (metric_width + 15)
        ]

        # Add each correlation
        for metric, corr in sorted_correlations:
            lines.append(f"{metric:<{metric_width}} {corr:>10.3f}")

        return "\n".join(lines)

    @staticmethod
    def format_churn_analysis(churned_ces, nonchurned_ces, churn_rates):
        """
        Format churn analysis results into a clear, readable table.

        Args:
            churned_ces (float): Average CES for churned clients
            nonchurned_ces (float): Average CES for non-churned clients
            churn_rates (dict): Dictionary of time periods and their churn rates
        """
        lines = [
            "Churn Analysis Results",
            "=" * 50,
            "\nCES Comparison",
            "-" * 30,
            f"{'Status':<20} {'Avg CES':>10}",
            "-" * 30,
            f"{'Churned':<20} {churned_ces:>10.2f}",
            f"{'Non-churned':<20} {nonchurned_ces:>10.2f}",
            "\nChurn Rate Timeline",
            "-" * 40,
            f"{'Time Period':<20} {'Rate':>10}",
            "-" * 40
        ]

        for period, rate in churn_rates.items():
            lines.append(f"{period:<20} {rate:>9.1f}%")

        return "\n".join(lines)

    @staticmethod
    def format_anova_results(test_name, f_stat, p_value, additional_info=None):
        """
        Format ANOVA test results in a clean table.

        Args:
            test_name (str): Name or description of the ANOVA test
            f_stat (float): F-statistic from the ANOVA
            p_value (float): p-value from the ANOVA
            additional_info (str, optional): Any additional context to display
        """
        lines = [
            f"ANOVA Results: {test_name}",
            "-" * 50
        ]

        if additional_info:
            lines.extend([additional_info, ""])

        lines.extend([
            f"{'Metric':<20} {'Value':>10} {'Significant':>12}",
            "-" * 50,
            f"{'F-statistic':<20} {f_stat:>10.3f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns':>12}",
            f"{'p-value':<20} {p_value:>10.4f}"
        ])

        return "\n".join(lines)

    @staticmethod
    def format_combination_analysis_header(combination_name, total_combinations=None, current_number=None):
        """Format the header for each combination analysis."""
        lines = []

        # Add progress indicator if we have total combinations
        if total_combinations and current_number:
            progress = f"[{current_number}/{total_combinations}]"
            lines.append(f"Analyzing Combination {progress}")
        else:
            lines.append("Analyzing Combination")

        lines.extend([
            "-" * 50,
            f"Features: {combination_name}",
            "-" * 50
        ])

        return "\n".join(lines)

    @staticmethod
    def format_data_overview(ces_data, churn_data):
        """
        Format the data overview section showing shapes, unique values, and date ranges.

        Args:
            ces_data (pd.DataFrame): CES survey data
            churn_data (pd.DataFrame): Churn data
        """
        ces_start = ces_data['Response_Timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
        ces_end = ces_data['Response_Timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
        churn_start = churn_data['Cancellation Scheduled Date'].min().strftime('%Y-%m-%d %H:%M:%S')
        churn_end = churn_data['Cancellation Scheduled Date'].max().strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            "Data Dimensions",
            "-" * 30,
            f"{'Dataset':<15} {'Records':>10} {'Columns':>10}",
            "-" * 30,
            f"{'CES':<15} {ces_data.shape[0]:>10,d} {ces_data.shape[1]:>10d}",
            f"{'Churn':<15} {churn_data.shape[0]:>10,d} {churn_data.shape[1]:>10d}",

            "\nUnique Identifiers",
            "-" * 40,
            f"{'Dataset':<15} {'Field':<15} {'Count':>10}",
            "-" * 40,
            f"{'CES':<15} {'ka':<15} {ces_data['ka'].nunique():>10,d}",
            f"{'Churn':<15} {'SiteID':<15} {churn_data['SiteID in CAA'].nunique():>10,d}",

            "\nDate Ranges",
            "-" * 75,
            f"{'Dataset':<15} {'From':<30} {'To'}",
            "-" * 75,
            f"{'CES':<15} {ces_start:<30} {ces_end}",
            f"{'Churn':<15} {churn_start:<30} {churn_end}"
        ]

        return "\n".join(lines)

    @staticmethod
    def format_cohort_comparison_results(results):
        """
        Format the cohort comparison results into a readable table.

        Args:
            results: List of dictionaries containing comparison results
        """
        lines = [
            "Cohort Comparison Analysis",
            "=" * 120,
            "\nPairwise T-Test Results",
            "-" * 120,
            f"{'Comparison':<15} {'Dates':<30} {'N1':>6} {'N2':>6} {'Mean 1':>8} {'Mean 2':>8} {'t-stat':>10} {'p-value':>10} {'Sig':>5}",
            "-" * 120,
        ]

        for r in results:
            sig_marker = '*' if r['significant'] else ''
            if r['p_value'] < 0.01:
                sig_marker = '**'
            if r['p_value'] < 0.001:
                sig_marker = '***'

            dates = f"({r['cohort1_start']} vs {r['cohort2_start']})"

            lines.append(
                f"{r['comparison']:<15} "
                f"{dates:<30} "
                f"{r['cohort1_n']:>6d} "
                f"{r['cohort2_n']:>6d} "
                f"{r['cohort1_mean']:>8.2f} "
                f"{r['cohort2_mean']:>8.2f} "
                f"{r['t_statistic']:>10.3f} "
                f"{r['p_value']:>10.4f} "
                f"{sig_marker:>5}"
            )

        # Add a legend for significance levels
        lines.extend([
            "",
            "Significance levels:",
            "* p < 0.05    ** p < 0.01    *** p < 0.001"
        ])

        return "\n".join(lines)

    @staticmethod
    def format_statistical_test(test_name, statistic, p_value, significance_levels=(0.05, 0.01, 0.001)):
        """
        Format statistical test results into a readable table row.

        Args:
            test_name (str): Name of the statistical test
            statistic (float): Test statistic value
            p_value (float): P-value from the test
            significance_levels (tuple): Tuple of significance levels for stars

        Returns:
            str: Formatted statistical test results
        """
        # Determine significance stars
        stars = ''
        for level in sorted(significance_levels, reverse=True):
            if p_value < level:
                stars = '*' * (len([l for l in significance_levels if l >= level]))
                break

        # Create formatted string
        lines = [
            "Statistical Test Results",
            "-" * 50,
            f"{'Test Type':<15} {'Statistic':>10} {'p-value':>10} {'Sig':>5}",
            "-" * 50,
            f"{test_name:<15} {statistic:>10.3f} {p_value:>10.3f} {stars:>5}"
        ]

        return "\n".join(lines)

    @staticmethod
    def format_efficiency_analysis(efficiency_stats, population_stats=None):
        """Format efficiency analysis results with means and population statistics."""
        lines = [
            "Efficiency Category Analysis",
            "-" * 65,
            f"{'Category':<25} {'Mean CES':>10} {'n':>6} {'CES %':>8} {'Pop %':>8}",
            "-" * 65
        ]

        total_n = sum(stats['n'] for stats in efficiency_stats.values())

        for category, stats in efficiency_stats.items():
            ces_pct = (stats['n'] / total_n) * 100
            pop_pct = (population_stats.get(category, 0) / sum(
                population_stats.values()) * 100) if population_stats else 0

            lines.append(
                f"{category:<25} {stats['mean']:>10.2f} {stats['n']:>6d} {ces_pct:>7.1f}% {pop_pct:>7.1f}%"
            )

        return "\n".join(lines)

    @staticmethod
    def format_category_distribution(category_counts, total=None, title="Category Distribution Analysis"):
        """
        Format category distribution into a readable table.

        Args:
            category_counts (dict): Dictionary of category names and their counts
            total (int, optional): Total number of items for percentage calculation
            title (str): Title for the distribution table

        Returns:
            str: Formatted category distribution table
        """
        if total is None:
            total = sum(category_counts.values())

        # Calculate dynamic spacing
        max_category_length = max(len(str(cat)) for cat in category_counts.keys())
        category_width = max(max_category_length, 25)

        # Create the formatted string
        lines = [
            title,
            "-" * (category_width + 25),
            f"{'Category':<{category_width}} {'Count':>8} {'%':>8}",
            "-" * (category_width + 25)
        ]

        # Add each category
        for category, count in category_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            lines.append(f"{category:<{category_width}} {count:>8d} {percentage:>7.1f}%")

        return "\n".join(lines)

    @staticmethod
    def format_cross_validation_results(cv_scores, title="Cross-validation Results"):
        """
        Format cross-validation scores into a readable table.

        Args:
            cv_scores (list): List of cross-validation scores
            title (str): Title for the results table

        Returns:
            str: Formatted cross-validation results table
        """
        lines = [
            title,
            "-" * 30,
            f"{'Fold':>4} {'Score':>10}",
            "-" * 30
        ]

        # Add each fold's score
        for i, score in enumerate(cv_scores, 1):
            lines.append(f"{i:>4} {score:>10.3f}")

        # Add summary statistics
        lines.extend([
            "-" * 30,
            f"{'Mean':>4} {np.mean(cv_scores):>10.3f}",
            f"{'Std':>4} {np.std(cv_scores):>10.3f}"
        ])

        return "\n".join(lines)

    @staticmethod
    def format_model_performance(metrics_dict, title="Model Performance Summary"):
        """
        Format model performance metrics into a readable table.

        Args:
            metrics_dict (dict): Dictionary of metric names and their values
            title (str): Title for the metrics table

        Returns:
            str: Formatted model performance table
        """
        # Calculate dynamic spacing
        max_metric_length = max(len(str(metric)) for metric in metrics_dict.keys())
        metric_width = max(max_metric_length, 15)

        lines = [
            title,
            "-" * (metric_width + 15),
            f"{'Metric':<{metric_width}} {'Score':>10}",
            "-" * (metric_width + 15)
        ]

        # Add each metric
        for metric, value in metrics_dict.items():
            lines.append(f"{metric:<{metric_width}} {value:>10.3f}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    pass