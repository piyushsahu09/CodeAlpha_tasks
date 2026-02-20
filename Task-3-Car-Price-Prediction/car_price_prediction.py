# TASK 3: Car Price Prediction with Machine Learning
# CodeAlpha Data Science Internship

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------
# 1. Load Dataset
# -----------------------------------
df = pd.read_csv("car_data.csv")

print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# -----------------------------------
# 2. Data Cleaning & Preprocessing
# -----------------------------------

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Drop non-numeric columns if any (like car name or brand)
numeric_df = df.select_dtypes(include=[np.number])

print("\nColumns used for modeling:")
print(numeric_df.columns)

# -----------------------------------
# 3. Feature Selection
# -----------------------------------
# Assuming last column is price
X = numeric_df.iloc[:, :-1]   # Features: horsepower, mileage, engine, etc.
y = numeric_df.iloc[:, -1]    # Target: car price

# -----------------------------------
# 4. Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# 5. Model Training (Regression)
# -----------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------------
# 6. Prediction
# -----------------------------------
y_pred = model.predict(X_test)

# -----------------------------------
# 7. Model Evaluation
# -----------------------------------
print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------------
# 8. Visualization
# -----------------------------------

# Actual vs Predicted Price
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Price")
plt.show()

# Feature importance (coefficients)
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

plt.figure(figsize=(10,5))
sns.barplot(x="Coefficient", y="Feature", data=coeff_df)
plt.title("Feature Impact on Car Price")
plt.show()

# -----------------------------------
# 9. Insights
# -----------------------------------
print("\nInsights:")
print("1. Features like horsepower, engine size, and mileage significantly impact car price.")
print("2. Higher horsepower and engine capacity generally increase car price.")
print("3. The regression model can help estimate fair car pricing in real-world markets.")
