# AI-phase
prediction of housing prices using machine learning
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your housing data into a DataFrame
# Replace 'data.csv' with the path to your dataset
data = pd.read_csv('naan m.csv')

# Separate features (X) and target variable (y)
X = data.drop('Price', axis=1)  # Replace 'Price' with the actual target column name
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Now you can use the trained model for predictions on new data
# For example, if you have a new set of features in 'X_new', you can predict the price as follows:
# new_predictions = model.predict(X_new)

