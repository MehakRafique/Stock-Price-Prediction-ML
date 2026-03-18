# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load stock data (Apple stock)
stock = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Step 2: Select features and target
# Features = Open, High, Low, Volume
X = stock[['Open', 'High', 'Low', 'Volume']]

# Target = Next day's Close price
y = stock['Close'].shift(-1)

# Remove last row (because it becomes NaN after shift)
X = X[:-1]
y = y[:-1]

# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Step 4: Train model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict values
y_pred = model.predict(X_test)

# Step 6: Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 7: Plot Actual vs Predicted Prices
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred, label="Predicted Price")
plt.legend()
plt.title("Actual vs Predicted Closing Prices")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()