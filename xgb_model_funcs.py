# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_data(filepath):
    """
    Load and preprocess data from a CSV file. This is a placeholder function.
    Actual preprocessing will depend on the specific requirements of the dataset.
    """
    data = pd.read_csv(filepath)
    # Replace this with actual preprocessing steps as necessary
    # Example: data['Price'] = data['Price'].replace('[\$,]', '', regex=True).astype(float)
    return data

def perform_grid_search(x_train, y_train):
    """
    Perform GridSearchCV to find the best hyperparameters for the XGBoost model.
    This function sets up a grid of parameters and uses cross-validation to find the best set.
    """
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.15],
        'min_child_weight': [1, 2, 3],
        'gamma': [0.0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'n_estimators': [100, 200, 300],
        'objective': ['reg:squarederror']
    }
    model = xgb.XGBRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

def train_and_evaluate_model(x_train, y_train, x_test, y_test, params):
    """
    General function to train and evaluate an XGBoost model.
    This function builds the model using given parameters, fits it on training data,
    makes predictions on the test set, and evaluates these predictions.
    """
    model = xgb.XGBRegressor(**params)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    r_squared = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    metrics = {
        "R-squared": r_squared,
        "Mean Absolute Error": mae,
        "Root Mean Squared Error": rmse
    }
    return model, metrics

# Definitions of each trial function including the unique parameters for different configurations.
def trial_1(x_train, y_train, x_test, y_test):
    """
    Trial 1: Baseline model with manually selected hyperparameters.
    This trial uses a specific set of parameters thought to provide a good starting point.
    """
    params = {
        'colsample_bytree': 0.9,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 3,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'subsample': 0.7
    }
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)

def trial_2(x_train, y_train, x_test, y_test):
    """
    Trial 2: Parameters adjusted based on GridSearchCV findings.
    This set of parameters reflects adjustments made following an extensive search
    for optimal values through cross-validation.
    """
    params = {
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'subsample': 0.8
    }
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)

def trial_3(x_train, y_train, x_test, y_test):
    """
    Trial 3: No condo data, treating "levels" as categorical.
    Unique handling of the dataset to focus on single-family homes.
    """
    params = {
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'subsample': 0.8
    }
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)

def trial_4(x_train, y_train, x_test, y_test):
    """
    Trial 4: Excludes condos and "levels" data, simplifying the feature set.
    This trial aims to test the impact of a more streamlined dataset on model performance.
    """
    params = {
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'subsample': 0.8
    }
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)

def trial_5(x_train, y_train, x_test, y_test):
    """
    Trial 5: No condos, no "levels", with SqFt outliers removed.
    Further refining the dataset by removing extreme SqFt values to see its impact on model accuracy.
    """
    params = {
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_child_weight': 1,
        'n_estimators': 300,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'subsample': 0.8
    }
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)
