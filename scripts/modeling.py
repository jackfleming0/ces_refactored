import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def prepare_data_for_modeling(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    X = X.fillna(X.mean())
    return X, y


def train_random_forest(X_train, y_train, **kwargs):
    model = RandomForestRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    logging.info(f"Model evaluation: RMSE={rmse}, R2={r2}")
    return rmse, r2


def cross_validate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    logging.info(f"Cross-validation RMSE scores: {rmse_scores}")
    return rmse_scores


if __name__ == "__main__":
    from feature_engineering import create_features
    from data_cleaning import clean_data
    from data_loading import load_config, load_data

    config = load_config()
    ces_data = load_data(config['data']['ces_data_path'])
    ces_data_clean = clean_data(ces_data, config['preprocessing']['date_columns'],
                                config['preprocessing']['date_formats'])
    ces_data_fe = create_features(ces_data_clean, config['cohorts'])

    feature_columns = ['leads_per_seat', 'account_age', 'ClientUser_Cnt', 'Client_#Logins', '30DayPageViews',
                       'Plan_Amt_Dec23']
    target_column = 'CES_Response_Value'
    X, y = prepare_data_for_modeling(ces_data_fe, feature_columns, target_column)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    rmse, r2 = evaluate_model(model, X_test, y_test)
    print(f"Test RMSE: {rmse}, R2: {r2}")
    # Save the model
    joblib.dump(model, '../outputs/models/random_forest_model.joblib')