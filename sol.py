# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data from sources
def load_data():
    # Load S&P Case-Schiller Home Price Index
    sp_data = pd.read_csv("/content/CSUSHPISA (1).csv")
    
    # Load data for other factors from chosen sources (e.g., FRED, Census)
    gdp_data = pd.read_csv("/content/GDPC1.csv")
    cpi_data = pd.read_csv("/content/CPIAUCSL.csv")
    # ... (load data for other factors)
    
    # Merge dataframes and handle missing values, outliers, etc.
    merged_data = pd.merge(sp_data, gdp_data, on="DATE")
    merged_data = pd.merge(merged_data, cpi_data, on="DATE")
    # ... (merge and handle data for other factors)
    
    return merged_data


# Feature selection and preparation
def prepare_features(data, required_features):
    # Check if all required features are present in the dataset
    missing_features = [feature for feature in required_features if feature not in data.columns]
    
    if not missing_features:
        X = data[required_features]
        y = data["home_price_index"]
        
        # Feature scaling (optional)
        # ... (implement feature scaling if needed, consider using StandardScaler)
        
        return X, y
    else:
        print(f"Error: Missing required features - {missing_features}")
        return None, None


# Train-test split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Model training and evaluation
def train_model(X_train, y_train):
    # Choose and implement your chosen model (e.g., Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    
    return model

# Predict home prices
def predict_prices(model, X_test, y_test):
    predicted_prices = model.predict(X_test)
    
    # Evaluate and compare predicted vs. actual prices
    mse = mean_squared_error(y_test, predicted_prices)
    print(f'Test Mean Squared Error: {mse}')
    
    if len(y_test) >= 2:
        r2 = r2_score(y_test, predicted_prices)
        print(f'Test R-squared: {r2}')
    else:
        print("Warning: Test set is too small for calculating R-squared.")

    # Visualize predicted vs. actual prices if needed
    # ... (matplotlib or other visualization libraries)

required_features = ["GDPC1", "CPIAUCSL", ...]

    # Load data
data = load_data()
    
    # Prepare features
X, y = prepare_features(data, required_features)
    
if X is not None:
  X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train model
model = train_model(X_train, y_train)
        
        # Predict home prices
predict_prices(model, X_test, y_test)
