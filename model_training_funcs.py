import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    """
    Load and clean the dataset from a CSV file.

    Parameters:
    filepath (str): The file path to the CSV data file.

    Returns:
    DataFrame: A cleaned pandas DataFrame with currency and numeric fields formatted.
    """
    data = pd.read_csv(filepath)
    columns_to_clean = ['List Price', 'Close Price', 'SqFt', 'LP$/SqFt', 'Close$/SqFt']
    for column in columns_to_clean:
        data[column] = data[column].replace('[\$,]', '', regex=True).replace(',', '', regex=True).astype(float)

    return data

def get_model_with_zip():
    """
    Train a linear regression model using all available data, incorporating zip codes as a training variable.
    
    Returns:
    tuple: Tuple containing the trained model, Root Mean Squared Error (RMSE), and R-squared value of the model.
    """
    data = load_and_clean_data('data.csv')
    data = data[data['Type of Home'] == 'Single Family']
    data = data.drop(columns=['MLS Area', 'Levels', 'Type of Home'])

    X = data.drop(columns=['Listing ID', 'St', 'Address', 'Close Price', 'Close Date', 'List Price', 'LP$/SqFt', 'Close$/SqFt'])
    y = data['Close Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mse)
    r_squared = model.score(X_test, y_test)
    
    return model, rmse, r_squared

def get_model_for_zip(zip_code):
    """
    Train a linear regression model for properties in a specific zip code.

    Parameters:
    zip_code (int): The zip code to filter the data by.

    Returns:
    tuple: Tuple containing the trained model, Root Mean Squared Error (RMSE), and R-squared value of the model.
    """
    data = load_and_clean_data('data.csv')
    data = data[(data['Zip Code'] == zip_code) & (data['Type of Home'] == 'Single Family')]
    data = data.drop(columns=['MLS Area', 'Levels', 'Type of Home'])

    X = data.drop(columns=['Listing ID', 'St', 'Address', 'Close Price', 'Close Date', 'List Price', 'LP$/SqFt', 'Close$/SqFt'])
    y = data['Close Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    rmse = np.sqrt(mse)
    r_squared = model.score(X_test, y_test)
    
    return model, rmse, r_squared
