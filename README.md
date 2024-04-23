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
features = ['# Beds', '# Full Baths', '# Half Baths', '# Garage Spaces', 'Year Built', 'Acres', 'SqFt', 'DOM', 'CDOM', 'Zip Code'] ![equation](https://github.com/SamuelSwafford/house-price-prediction-model/assets/52751074/8468c39d-5f98-4bf5-b68e-616669e6af80)

Linear regression is a statistical method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.
Creating a linear model for each zip code was more accurate for some zip codes and less accurate for others.
![r](https://github.com/SamuelSwafford/house-price-prediction-model/assets/52751074/565fc968-8bcb-4486-9c8c-1173a158a49d)

## eXtreme Gradient Boosting
Gradient boosting is a machine learning ensemble technique that combines the predictions of multiple weak learners (decision trees) sequentially.
The predicting housing prices involves dealing with complex, high-dimensional data. Therefore, XGBoost is well-suited for handling data with mixed feature types and
captures nonlinear relationships between features and target variables to model intricate patterns in housing price data.
![Screenshot 2024-04-22 194454](https://github.com/SamuelSwafford/house-price-prediction-model/assets/52751074/93764c59-0705-47a5-b934-b45abe0f84ff)
## GridSearchCV (Fine Tuning)
GridSearchCV searches through all the parameter combinations provided in a grid, performing cross-validation on each combination to evaluate its performance.
The performing hyperparameter turned a range of values like learning rate, maximum depth, the number of estimators, etc.
Target encoding was utilized to convert zip code and home type to numeric, capture the relationships between categorical values and the target (closing price), and dropped unhelpful variables (# of levels and condos). 
![Screenshot 2024-04-22 194249](https://github.com/SamuelSwafford/house-price-prediction-model/assets/52751074/c21b81c2-d19d-4b70-9853-827408cdb652)
## Ensemble Model
An Ensemble Model leverages the strengths of each model. In this project, 4 Machine Learning Techniques were ustilized: Random Forest, Linear Regression, and XGBoost. By creating the Ensemble Model, it helps generate a more accurate prediction overall.
![Ensemble Model](https://github.com/SamuelSwafford/house-price-prediction-model/assets/52751074/9fae57c9-5b23-4589-9329-f3441e36251a)

## Zillow House Comparison to Models
Identified Two Zillow Listings located in Pflugerville, Texas to compare to our models
- 13309 Clerk St
Beds: 4
Garage Spaces: 2
Year Built: 2019
Acres: 0.14
SqFt 2603
CDOM: 2 (based on Zillow listing)
Total Baths: 2.5
Zip Code: "78660" (Pflugerville, TX)
Home Type: "Single Family"
List Price: $450,000
Linear Regression Model Predicted Price = $469,893.38
Ensemble Model Model Predicted Price = $464,163.42
- 1209 Vapor Dr
Beds 6
Garage Spaces 8
Year Built 1978
Acres 1.1
SqFt 2863
CDOM 2 (based on Zillow listing)
Total Baths 4
Zip Code "78660" (Pflugerville, TX)
Home Type "Single Family"
List Price $899,999
Line Regression Model Predicted Price = $779,380.94
Ensemble Model Predicted Price = $741,993.42

## Conclusion
Overall, sentiment analysis serves as a valuable tool for investors, policymakers, and real estate professionals to understand market sentiment, identify trends, and make informed decisions in the dynamic housing market landscape.
Predictive Model Analysis in real estate helps to optimize investment decisions and improve property valuation accuracy.
Using our models, we can come up with a reasonable estimate for a home seller’s potential property valuation within Austin, Texas.

## Data Summary 
For Sentiment Analysis, the data analyzed came from News Articles: News API and The New York Times.
Data utilized to Predict Home Sales for the Machine Learning was obtained from Austin Board of Realtors - Multiple Listing Service.
We utilized Sales from the last 6 months that are marked as “Closed”. Sales from 6 months contained a lot of data, so we narrowed it down to Single Family Homes & Condominiums
from the following 9 Zip Codes: 78746, 78758, 78744, 78664, 78660, 78701, 78620, 78642, 78666. These zip codes are from Austin and it's sorrounding areas.

![ABOR_MLS](https://github.com/SamuelSwafford/house-price-prediction-model/assets/52751074/f7928e2f-aac0-4512-b79a-ae2cd6e15d1c)
## Citations
MLS Compliance Resources Kit. Austin Board of REALTORS®. (n.d.). https://www.abor.com/ACTNow 
Home. Austin Board of REALTORS®. (n.d.). https://www.abor.com/ 

