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

# Filter the dataset for a specific MLS Area, e.g., '1A'
data_specific_area = data[data['MLS Area'] == 'RRE']

# Drop the 'MLS Area' as it is not needed anymore after filtering
data_specific_area = data_specific_area.drop(columns=['MLS Area', 'Levels'])  # Now also removing 'Levels'

# Define the features and target variable for the filtered dataset
X_specific_area = data_specific_area.drop(columns=['Listing ID', 'St', 'Address', 'Close Price', 'Close Date', 'List Price', 'LP$/SqFt', 'Close$/SqFt'])
y_specific_area = data_specific_area['Close Price']

# Split the dataset into training and testing sets
X_train_specific_area, X_test_specific_area, y_train_specific_area, y_test_specific_area = train_test_split(X_specific_area, y_specific_area, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model_specific_area = LinearRegression()
model_specific_area.fit(X_train_specific_area, y_train_specific_area)

# Make predictions and evaluate the model
predictions_specific_area = model_specific_area.predict(X_test_specific_area)
mse_specific_area = mean_squared_error(y_test_specific_area, predictions_specific_area)
rmse_specific_area = np.sqrt(mse_specific_area)
r_squared = model_specific_area.score(X_test_specific_area, y_test_specific_area)

print(f'Root Mean Squared Error: {rmse_specific_area}')
print(f'R-squared: {r_squared}')
