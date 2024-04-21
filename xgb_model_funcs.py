# Import necessary libraries
import pandas as pd
import xgboost as xgb
import json
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

def perform_grid_search(x_train, y_train, filename='grid_search_results.json'):
    """
    Perform GridSearchCV to find the best hyperparameters for the XGBoost model.
    This function sets up a grid of parameters, uses cross-validation to find the best set,
    and saves the results to a JSON file.
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
    best_params = grid_search.best_params_
    
    # Save best parameters to JSON file
    with open(filename, 'w') as f:
        json.dump(best_params, f)
    
    return grid_search.best_estimator_, best_params

def load_params_from_file(filename='grid_search_results.json'):
    """
    Load the best hyperparameters from a JSON file.
    """
    with open(filename, 'r') as f:
        params = json.load(f)
    return params

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

def trial_1(x_train, y_train, x_test, y_test):
    """
    Trial 1: Baseline model with manually selected hyperparameters.
    """
    params = load_params_from_file()  # Load parameters from file
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)

def trial_2(x_train, y_train, x_test, y_test):
    """
    Trial 2: Parameters adjusted based on GridSearchCV findings.
    """
    params = load_params_from_file()  # Load parameters from file
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)

def trial_3(x_train, y_train, x_test, y_test):
    """
    Trial 3: No condo data, treating "levels" as categorical.
    """
    params = load_params_from_file()  # Load parameters from file
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)

def trial_4(x_train, y_train, x_test, y_test):
    """
    Trial 4: Excludes condos and "levels" data, simplifying the feature set.
    """
    params = load_params_from_file()  # Load parameters from file
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)

def trial_5(x_train, y_train, x_test, y_test):
    """
    Trial 5: No condos, no "levels", with SqFt outliers removed.
    """
    params = load_params_from_file()  # Load parameters from file
    return train_and_evaluate_model(x_train, y_train, x_test, y_test, params)
