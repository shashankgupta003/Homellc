
Publicly Available Data for US Home Price Factors
Here are some publicly available data sources for key factors that influence US home prices nationally:

Economic Factors:

Federal Reserve Bank Economic Data: [fred.stlouisfed.org]
GDP (Gross Domestic Product): [fred.stlouisfed.org/series/GDPC1]
Consumer Price Index (CPI): [fred.stlouisfed.org/series/CPIAUCSL]
Mortgage Rates: [fred.stlouisfed.org/series/MORTGAGE30US]
Unemployment Rate: [fred.stlouisfed.org/series/UNRATE]
Consumer Confidence Index: [www.theconferenceboard.org/data-pdf-consumerconfidenceindex.aspx]
Bureau of Labor Statistics: [bls.gov]
Average Wages: [bls.gov]
Housing Starts: [bls.gov]
Demographic Factors:

US Census Bureau: [census.gov]
Population by Age and Sex: [census.gov]
Household Income: [census.gov]
Education Level: [census.gov]
Educational Attainment: [census.gov]
Gallup Poll: [news.gallup.com]
Home Buying Sentiment: [news.gallup.com]
Housing Market Factors:

National Association of Realtors: [nar.realtor]
Existing Home Sales: [nar.realtor]
New Home Sales: [nar.realtor]
Housing Inventory: [nar.realtor]
Median Home Price: [nar.realtor]
Freddie Mac: [freddiemac.com]
Mortgage Rates: [freddiemac.com]
Other Resources:

S&P Case-Schiller Home Price Index: [fred.stlouisfed.org/series/CSUSHPISA]
Zillow Home Value Index: [zillow.com]
Trulia HPI: [trulia.com]
Data Science Model
Once you have collected the relevant data, you can build a data science model to explain how these factors impacted home prices over the last 20 years. Here are some possible approaches:

Regression Model:

Develop a linear regression or other regression model to predict home prices based on the collected factors. This model can provide insights into the relative importance of each factor and how changes in these factors affect home prices.
You can use libraries like scikit-learn in Python to implement these models.
Machine Learning Model:

Explore machine learning models like random forests, gradient boosting, or neural networks. These models can capture complex relationships between the factors and home prices, potentially providing more accurate predictions than simpler models.
Libraries like TensorFlow and PyTorch can be used for building these models.
Time Series Model:

Consider using a time series forecasting model like ARIMA or LSTM to predict future home prices based on historical data.
Libraries like statsmodels and PyTorch can be used for time series forecasting.
Model Evaluation:

Evaluate the performance of your model using metrics like Mean Squared Error (MSE) or R-squared. This helps assess how well the model predicts actual home prices.
Analyze the model's coefficients or feature importance to understand which factors have the most significant impact on home prices.
Additional Considerations
Data Cleaning and Preprocessing: Ensure your data is clean and preprocessed before feeding it into your model. This might involve handling missing values, outliers, and inconsistencies.
Feature Engineering: Create new features based on existing data to capture relevant information for your model.
Model Tuning: Tune the hyperparameters of your model to optimize its performance.
Visualization: Visualize the relationships between the factors and home prices to gain insights and interpret your model's results.