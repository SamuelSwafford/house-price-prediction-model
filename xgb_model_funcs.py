# Import necessary libraries
import pandas as pd
import xgboost as xgb
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_and_clean_data(filepath, include_condos=True, treat_levels_as_categorical=False, remove_sqft_outliers=False):
    """
    Load and clean the dataset from a CSV file.
    """
    data = pd.read_csv(filepath)

    # Columns to clean for financial figures
    columns_to_clean = ['List Price', 'Close Price', 'SqFt', 'LP$/SqFt', 'Close$/SqFt']
    for column in columns_to_clean:
        data[column] = data[column].replace('[\$,]', '', regex=True).replace(',', '', regex=True).astype(float)

    # Dropping columns that may not be relevant or are textual descriptions
    columns_to_drop = ['St', 'MLS Area', 'Address', 'Close Date', 'Type of Home']
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Conditionally exclude condo data
    if not include_condos and 'Home Type' in data.columns:
        data = data[data['Home Type'] != 'Condo']

    # Handling 'Levels' if it's categorical
    if treat_levels_as_categorical and 'Levels' in data.columns:
        levels_mapping = {'One': 1, 'Two': 2, 'Three': 3}
        data['Levels'] = data['Levels'].map(levels_mapping)

    # Remove outliers in 'SqFt' if required
    if remove_sqft_outliers and 'SqFt' in data.columns:
        mean_sqft = data['SqFt'].mean()
        std_sqft = data['SqFt'].std()
        data = data[(data['SqFt'] > (mean_sqft - 3 * std_sqft)) & (data['SqFt'] < (mean_sqft + 3 * std_sqft))]

    return data

def prepare_training_testing_data(filepath, test_size=0.2, random_state=42, include_condos=True, treat_levels_as_categorical=False, remove_sqft_outliers=False):
    """
    Load data, clean it, and split into training and testing datasets.
    """
    data = load_and_clean_data(filepath, include_condos, treat_levels_as_categorical, remove_sqft_outliers)
    y = data.pop('Close Price')  # Assuming 'Close Price' is the target variable
    X = data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return x_train, y_train, x_test, y_test

def perform_grid_search(x_train, y_train, filename='grid_search_results.json'):
    """
    Perform GridSearchCV to find the best hyperparameters for the XGBoost model.
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
