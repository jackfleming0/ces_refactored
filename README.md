Here's the spreadsheet to use! https://docs.google.com/spreadsheets/d/1xqxoQM9vrN_CZR510my7S8lbCdGOqo0ohygdUSBzhHM/edit?gid=0#gid=0

Instructions are in there for how to do the data gathering side. the below is about the code portion.

btw: When you add in the ad spend data, be sure to delete any non-header-lines up top. otherwise it won't run well.

---------------------------------------------

# CES Analysis System

## Overview

This set of python files provides comprehensive analysis capabilities for Customer Effort Score (CES) data, integrating with advertising spend metrics, churn prediction, and various business metrics. It's designed to be modular, extensible, and provide robust statistical analysis and visualization capabilities.

## Table of Contents

1. Prerequisites
2. Installation
3. Project Structure
4. Configuration
5. Usage
6. Core Components
7. Analysis Capabilities
8. Visualization
9. Common Issues & Troubleshooting
10. Maintenance Notes

### Prerequisites

```
Python 3.8+

Required packages (install via pip):
Copypandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
plotly
shap
PyYAML
networkx
scipy
```


### Installation

1. Clone this repository
2. Install required packages:
3. `bash Copypip install -r requirements.txt`

Verify the config.yaml file is properly configured for your environment

Ensure your data files are in the correct location as specified in config.yaml

### Project Structure
```
Copyproject/
├── config/
│   └── config.yaml        # Configuration settings
├── data/
│   ├── ces_data.csv      # Main CES data
│   └── churned_clients.csv # Churn data
├── outputs/
│   ├── analyses/         # Analysis outputs
│   ├── figures/         # Generated visualizations
│   └── models/          # Saved models
└── src/
    ├── main.py          # Main orchestration script
    ├── data_loading.py  # Data loading utilities
    ├── data_cleaning.py # Data cleaning operations
    ├── eda.py          # Exploratory data analysis
    ├── churn_analysis.py # Churn analysi
    ├── feature_engineering.py # Feature creation
    ├── modeling.py     # Machine learning models
    ├── stability_analysis.py # Stability analysis
    ├── statistical_tests.py # Statistical testing
    ├── utils.py        # Utility functions
    └── visualization.py # Visualization functions
```

### Configuration

The system is configured via config.yaml. Key configuration sections:
```
yaml

Copydata:
  ces_data_path: Path to CES data
  
preprocessing:
  date_columns: Columns to parse as dates
  date_formats: Accepted date formats

cohorts:
  Group definitions and date ranges

columns_to_bin:
  Numerical columns to bin and number of bins

modeling:
  feature_columns: Features for machine learning
```

## Usage

This section will show a few samples of usage as a way to demonstrate what each file is used for. 

### Basic Usage

Update configuration in config.yaml

Run the main analysis:
`python main.py`


### Common Analysis Tasks
#### Running CES Analysis

```
from data_loading import load_config, load_data
from data_cleaning import clean_data
from feature_engineering import create_features

# Load and prepare data
config = load_config()
ces_data = load_data(config['data']['ces_data_path'])
ces_data_clean = clean_data(ces_data, config['preprocessing']['date_columns'],
                           config['preprocessing']['date_formats'])
ces_data_fe = create_features(ces_data_clean, config['cohorts'])

# Run basic analysis
from eda import analyze_ces_distributions
analyze_ces_distributions(ces_data_fe)
```

#### Generating Visualizations
```
from visualization import generate_all_visualizations
generate_all_visualizations(ces_data_fe, model, config, combinatorial_results)
```

## Core Components of this project
### Data Loading (`data_loading.py`)

Handles data ingestion and initial preprocessing. Key functions:

* `load_config()`: Loads configuration from YAML
* `load_data()`: Loads CSV data files
* `preprocess_data()`: Initial data preprocessing

### Data Cleaning (`data_cleaning.py`)
Handles data cleaning operations:

* Drops missing CES values
* Parses dates
* Handles missing values

### Feature Engineering (`feature_engineering.py`)
Creates calculated variables and features:

* Assigns cohorts
* Calculates account age
* Processes ad spend data
* Creates binned variables

## Analysis Modules
### EDA (`eda.py`)
Exploratory Data Analysis functions:

* CES distribution analysis
* Correlation analysis
* Content pages analysis
* Ad spend analysis

### Churn Analysis (`churn_analysis.py`)
Analyzes relationship between CES and churn:

* Propensity Score Matching
* Logistic regression
* Visualization of churn patterns

### Statistical Tests (`statistical_tests.py`)
Statistical testing capabilities:

* T-tests between groups
* ANOVA testing
* Tukey's HSD test

### Stability Analysis (`stability_analysis.py`)
Analyzes CES stability across different dimensions:

* Combinatorial analysis
* Elasticity analysis
* Trend analysis

## Analysis Capabilities
### CES Analysis

* Distribution analysis
* Cohort comparison
* Time-based trends
* Categorical analysis

### Ad Spend Analysis

* Correlation with CES
* Efficiency categories
* Regional patterns
* Spend bracket analysis

### Churn Analysis

* Propensity score matching
* Predictive modeling
* Churn rate analysis
* CES-churn relationship

### Statistical Analysis

* Group comparisons
* ANOVA testing
* Trend analysis
* Stability metrics

## Visualization
There's some pretty robust visualization stuff, though the styling could certainly use some work :)  (`visualization.py`):

* Distribution plots
* Time series analysis
* Correlation heatmaps
* Box plots
* Radar charts
* Sankey diagrams

## Common Issues & Troubleshooting
### Data Loading Issues

* Verify file paths in config.yaml
* Check CSV file encoding
* Ensure date formats match configuration
* MAKE SURE THAT CONFIG.YAML IS UPDATED WITH EACH COHORT DATES
* You gotta make sure that the site ID in the churned clients sheet is set to Format>Number>Automatic vs whatever it comes in as. not sure the deal tbh 

### Visualization Errors

* Check for missing data in plot variables
* Verify categorical variables are properly encoded
* Ensure sufficient data for statistical tests


# Contact
For questions or issues, please contact [jack.k.fleming@gmail.com].  I don't work here anymore but i'll probably have time to help! 