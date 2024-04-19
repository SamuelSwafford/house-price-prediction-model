import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import numpy as np

# Load the data
data = pd.read_csv('data.csv')

# Data cleaning: Convert currency to floats and handle numeric fields properly
columns_to_clean = ['List Price', 'Close Price', 'SqFt', 'LP$/SqFt', 'Close$/SqFt']
for column in columns_to_clean:
    data[column] = data[column].replace('[\$,]', '', regex=True).replace(',', '', regex=True).astype(float)

# Filter the dataset for only Single Family homes
data_filtered = data[data['Type of Home'] == 'Single Family']

# Drop the 'MLS Area' and 'Levels' as they are not needed anymore after filtering
data_filtered = data_filtered.drop(columns=['MLS Area', 'Levels', 'Type of Home'])

# Define the features and target variable for the filtered dataset
X_filtered = data_filtered.drop(columns=['Listing ID', 'St', 'Address', 'Close Price', 'Close Date', 'List Price', 'LP$/SqFt', 'Close$/SqFt'])
y_filtered = data_filtered['Close Price']

# Include Zip Code as a numeric variable in the model
X_filtered['Zip Code'] = data_filtered['Zip Code'].astype(float)  # Ensure Zip Code is treated as numeric

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r_squared = model.score(X_test, y_test)

print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r_squared}')
