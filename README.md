# Predicting Housing Prices and exploring current market Sentiments

## Project Goals
Sentiment Analysis: 
The purpose of this section is to analyze sentiment in the housing market using data extracted from news articles via the News API.
Sentiment analysis aims to understand the overall emotional tone of the news articles regarding the housing market.

Predicting Housing Prices Using Machine Learning Techniques:
The focus of this section is on predicting housing prices in the Austin Market using the gradient boosting and linear regression models.
Predictive modeling allows stakeholders to forecast Austin's housing market trends and make informed decisions regarding buying, selling, or investing in properties.

## Sentiment Analysis and Topic Modeling

## Linear Regression Models for Zip Codes


## eXtreme Gradient Boosting
Gradient boosting is a machine learning ensemble technique that combines the predictions of multiple weak learners (decision trees) sequentially.
The predicting housing prices involves dealing with complex, high-dimensional data. Therefore, XGBoost is well-suited for handling data with mixed feature types and
captures nonlinear relationships between features and target variables to model intricate patterns in housing price data.

## GridSearchCV (Fine Tuning)
GridSearchCV searches through all the parameter combinations provided in a grid, performing cross-validation on each combination to evaluate its performance.
The performing hyperparameter turned a range of values like learning rate, maximum depth, the number of estimators, etc.
Target encoding was utilized to convert zip code and home type to numeric, capture the relationships between categorical values and the target (closing price), and dropped unhelpful variables (# of levels and condos). 

## Ensemble Model
An Ensemble Model leverages the strengths of each model. In this project, 4 Machine Learning Techniques were ustilized: Random Forest, Linear Regression, and XGBoost. By creating the Ensemble Model, it helps generate a more accurate prediction overall.

## Data Summary 
For Sentiment Analysis, the data analyzed came from News Articles: News API and The New York Times.
Data utilized to Predict Home Sales for the Machine Learning was obtained from Austin Board of Realtors - Multiple Listing Service.
We utilized Sales from the last 6 months that are marked as “Closed”. Sales from 6 months contained a lot of data, so we narrowed it down to Single Family Homes & Condominiums
from the following 9 Zip Codes: 78746, 78758, 78744, 78664, 78660, 78701, 78620, 78642, 78666. These zip codes are from Austin and it's sorrounding areas.

## Citations
MLS Compliance Resources Kit. Austin Board of REALTORS®. (n.d.). https://www.abor.com/ACTNow 
Home. Austin Board of REALTORS®. (n.d.). https://www.abor.com/ 
