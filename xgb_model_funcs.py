# Import necessary libraries
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_and_evaluate_model(x_train, y_train, x_test, y_test, params):
    """
    General function to train and evaluate an XGBoost model.
    """
    # Build the XGBRegressor model
    model = xgb.XGBRegressor(**params)
    
    # Fit the model to the training data
    model.fit(x_train, y_train)
    
    # Make predictions
    predictions = model.predict(x_test)
    
    # Evaluate the model
    r_squared = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    # Dictionary of evaluation metrics
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
    Further refining the dataset by removing extreme SqFt values.
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
