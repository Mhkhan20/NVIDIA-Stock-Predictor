import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv('C:/Users/mhass/OneDrive/Desktop/Education/Projects/DataAnalytics/data/Nvidia_Stock_Prices.csv')

# Select features and targets
features = df[['Open', 'Close', 'Volume']]  # Common features to predict High and Low
y_high = df['High']  # Target for the High model
y_low = df['Low']    # Target for the Low model

# Split the data into training and testing sets for both targets
X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(features, y_high, test_size=0.2, random_state=42)
X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(features, y_low, test_size=0.2, random_state=42)

# Create and train the linear regression models
model_high = LinearRegression()
model_low = LinearRegression()

model_high.fit(X_train_high, y_train_high)
model_low.fit(X_train_low, y_train_low)

# Make predictions using the testing set for both models
y_pred_high = model_high.predict(X_test_high)
y_pred_low = model_low.predict(X_test_low)

# Evaluate the models
mse_high = mean_squared_error(y_test_high, y_pred_high)
mse_low = mean_squared_error(y_test_low, y_pred_low)

r2_high = r2_score(y_test_high, y_pred_high)
r2_low = r2_score(y_test_low, y_pred_low)

print(f"High Model - Mean Squared Error: {mse_high}, R^2 Score: {r2_high}")
print(f"Low Model - Mean Squared Error: {mse_low}, R^2 Score: {r2_low}")

# Visualization of the results
plt.figure(figsize=(12, 6))

# High value predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test_high, y_pred_high, alpha=0.5, color='blue')
plt.title('Actual vs. Predicted High Prices')
plt.xlabel('Actual High Prices')
plt.ylabel('Predicted High Prices')
plt.plot([y_test_high.min(), y_test_high.max()], [y_test_high.min(), y_test_high.max()], color='red')

# Low value predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test_low, y_pred_low, alpha=0.5, color='green')
plt.title('Actual vs. Predicted Low Prices')
plt.xlabel('Actual Low Prices')
plt.ylabel('Predicted Low Prices')
plt.plot([y_test_low.min(), y_test_low.max()], [y_test_low.min(), y_test_low.max()], color='red')

plt.tight_layout()
plt.show()
