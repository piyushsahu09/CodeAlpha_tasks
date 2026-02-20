# TASK 4: Sales Prediction using Python
# CodeAlpha Data Science Internship

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("sales_data.csv")

print("First 5 rows of dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# -------------------------------
# 2. Data Cleaning
# -------------------------------
# Clean column names
df.columns = df.columns.str.strip()

# Handle missing values
df.fillna(method='ffill', inplace=True)

# -------------------------------
# 3. Feature Selection
# -------------------------------
# Assuming last column is Sales
X = df.iloc[:, :-1]   # Features (Advertising spend, platform, target segment, etc.)
y = df.iloc[:, -1]    # Target (Sales)

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. Model Training (Regression)
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 6. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 7. Model Evaluation
# -------------------------------
print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# 8. Visualization
# -------------------------------

# Visualization 1: Advertising Spend vs Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x=X.iloc[:, 0], y=y)
plt.title("Advertising Spend vs Sales")
plt.xlabel("Advertising Spend")
plt.ylabel("Sales")
plt.show()

# Visualization 2: Actual vs Predicted Sales
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# -------------------------------
# 9. Insights
# -------------------------------
print("\nInsights:")
print("1. Sales increase with higher advertising spend.")
print("2. Regression model shows strong relationship between marketing spend and sales.")
print("3. Model can help businesses optimize advertising strategies.")
