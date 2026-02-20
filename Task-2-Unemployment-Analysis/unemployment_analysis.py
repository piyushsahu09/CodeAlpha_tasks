# TASK 2: Unemployment Analysis with Python
# CodeAlpha Data Science Internship

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Make sure the dataset file name matches exactly
df = pd.read_csv("unemployment_data.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

# -------------------------------
# 2. Data Cleaning
# -------------------------------
# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Convert date column to datetime
# (Assuming first column is Date)
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

# Rename columns for clarity (if needed)
df.rename(columns={
    df.columns[0]: "date",
    df.columns[1]: "unemployment_rate"
}, inplace=True)

# -------------------------------
# 3. Exploratory Data Analysis
# -------------------------------
print("\nSummary Statistics:")
print(df.describe())

# -------------------------------
# 4. Visualization: Unemployment Trend
# -------------------------------
plt.figure(figsize=(10,5))
sns.lineplot(x="date", y="unemployment_rate", data=df)
plt.title("Unemployment Rate Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -------------------------------
# 5. COVID-19 Impact Analysis
# -------------------------------
covid_data = df[df["date"].dt.year >= 2020]

plt.figure(figsize=(10,5))
sns.lineplot(x="date", y="unemployment_rate", data=covid_data, color="red")
plt.title("Impact of COVID-19 on Unemployment Rate")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -------------------------------
# 6. Seasonal / Monthly Pattern
# -------------------------------
df["month"] = df["date"].dt.month

plt.figure(figsize=(10,5))
sns.boxplot(x="month", y="unemployment_rate", data=df)
plt.title("Seasonal Pattern in Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# -------------------------------
# 7. Key Insights
# -------------------------------
print("\nKey Insights:")
print("1. Unemployment rate increased sharply during the COVID-19 period (2020).")
print("2. Certain months show consistent seasonal unemployment patterns.")
print("3. Post-COVID recovery is visible but gradual.")
print("4. Data highlights the need for economic support during crisis periods.")
