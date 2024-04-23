# Predicting Housing Prices and exploring current market Sentiments

## Project Goals
Sentiment Analysis: 
The purpose of this section is to analyze sentiment in the housing market using data extracted from news articles via the News API.
Sentiment analysis aims to understand the overall emotional tone of the news articles regarding the housing market.

Predicting Housing Prices Using Machine Learning Techniques:
The focus of this section is on predicting housing prices in the Austin Market using the gradient boosting and linear regression models.
Predictive modeling allows stakeholders to forecast Austin's housing market trends and make informed decisions regarding buying, selling, or investing in properties.

## Sentiment Analysis and Topic Modeling

## Methods and Application

Natural Language Processing (NLP) Libraries: 
NLTK is used for tasks like sentiment analysis, and topic modeling. The sentiment score for each article's content is computed using VADER's compound score, which ranges from -1 (most negative) to 1 (most positive). LDA was used for topic modeling, the LDA can assist in organizing and summarizing large amounts of text data, making it easier to navigate and extract valuable insights.

Text Vectorization: Process of converting text data into numerical vectors that machine learning algorithms can process. Text vectorization is performed using CountVectorizer,this converts preprocessed text data into a document-term matrix.

Word Cloud Visualization: Exploratory data analysis tool to highlight frequently occurring terms.

# Sentiment Analysis 

Sentiment analysis can have a significant impact on the housing market by providing valuable insights to investors, policymakers, and real estate professionals. Here's how each group can leverage sentiment analysis to make informed decisions:

## Investors:
Market Sentiment: Investors can use sentiment analysis to gauge the overall sentiment of the housing market. Positive sentiment may indicate a strong market with potential for growth, while negative sentiment may signal caution or a potential downturn.
Risk Management: By analyzing sentiment trends, investors can identify potential risks and adjust their investment strategies accordingly. For example, if sentiment is turning negative, investors may choose to diversify their portfolios or reduce exposure to the housing market.
Timing Investments: Sentiment analysis can help investors identify opportune times to enter or exit the market. Positive sentiment may signal a favorable buying opportunity, while negative sentiment may indicate a time to sell or hold off on new investments.
## Policymakers:
Policy Decisions: Policymakers can use sentiment analysis to inform their decisions regarding housing policies and regulations. Understanding market sentiment can help policymakers anticipate the impact of proposed policies and make adjustments to support a stable and sustainable housing market.
Economic Indicators: Sentiment analysis can complement traditional economic indicators by providing real-time insights into consumer confidence and market sentiment. Policymakers can use this information to fine-tune monetary or fiscal policies to support economic growth and stability.
## Real Estate Professionals:
Market Analysis: Real estate professionals can use sentiment analysis to conduct market analysis and assess the demand and supply dynamics in specific regions or segments of the housing market. Positive sentiment may indicate high demand and rising prices, while negative sentiment may signal a softening market.

# Sample of Sentiment and assigned scores
![image](https://github.com/SamuelSwafford/house-price-prediction-model/assets/148410176/7b1ae4ce-e106-4291-9b1c-82b1e035b507)

Each article are assigned sentiment scores and reflects whether each artice are positive, negative or neutral.

![image](https://github.com/SamuelSwafford/house-price-prediction-model/assets/148410176/3ff24f6b-6a24-4294-a59c-f39c8369f972)

The bar chart portrays sentiments derived from a decade-long sample of news articles, predominantly reflecting negative sentiments. Several headlines contributing to this trend are outlined.

![image](https://github.com/SamuelSwafford/house-price-prediction-model/assets/148410176/4b5d4f39-6f96-402d-95d2-36eb6ab044ec)

The presented bar chart illustrates the prevailing sentiment in the market, indicating a predominantly positive tone during the observed period. This positivity can be attributed to several contributing factors:

Increased construction of new homes in March, with single-family housing starts rising by 2.7 percent to 861,000.
Relative stability in month-to-month house prices over recent months.
A downward trend in mortgage rates compared to previous highs.
Furthermore, optimism is evident as Americans display increased confidence in both buying and selling homes, as indicated by the Fannie Mae Home Purchase Sentiment Index.

## Word Cloud

![image](https://github.com/SamuelSwafford/house-price-prediction-model/assets/148410176/080f8550-7764-416f-b462-5a9f56adfd48)



A word cloud is a visual representation of text data in which the size of each word indicates its frequency or importance. It's a popular and intuitive way to summarize and visualize textual information, making it easier to identify the most prominent words or themes within a body of text. A few of the prominent words listed in News article reviews are rate, home, price etc.


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

