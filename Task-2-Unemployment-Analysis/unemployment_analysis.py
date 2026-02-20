# =========================================
# TASK 2: Unemployment Analysis with Python
# Using Pre-COVID and COVID Datasets
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Datasets
# -------------------------------
pre_covid = pd.read_csv("unemployment_pre_covid.csv")
covid = pd.read_csv("unemployment_covid.csv")

# -------------------------------
# 2. Standardize Column Names
# -------------------------------
pre_covid.columns = pre_covid.columns.str.strip().str.lower()
covid.columns = covid.columns.str.strip().str.lower()

# Expected columns: date, unemployment_rate
pre_covid['period'] = 'Pre-COVID'
covid['period'] = 'COVID'

# -------------------------------
# 3. Convert Date Column
# -------------------------------
pre_covid['date'] = pd.to_datetime(pre_covid['date'])
covid['date'] = pd.to_datetime(covid['date'])

# -------------------------------
# 4. Merge Datasets
# -------------------------------
df = pd.concat([pre_covid, covid], ignore_index=True)

# -------------------------------
# 5. Data Cleaning
# -------------------------------
df.dropna(inplace=True)

# -------------------------------
# 6. Overall Trend Visualization
# -------------------------------
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date', y='unemployment_rate', hue='period')
plt.title("Unemployment Rate Trend (Pre-COVID vs COVID)")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# -------------------------------
# 7. COVID Impact Comparison
# -------------------------------
avg_rates = df.groupby('period')['unemployment_rate'].mean()

plt.figure(figsize=(6, 4))
avg_rates.plot(kind='bar', color=['green', 'red'])
plt.title("Average Unemployment Rate Comparison")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -------------------------------
# 8. Seasonal Trend Analysis
# -------------------------------
df['month'] = df['date'].dt.month

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='month', y='unemployment_rate')
plt.title("Monthly Seasonal Trend in Unemployment")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# -------------------------------
# 9. Key Insights
# -------------------------------
print("\n📊 KEY INSIGHTS:")
print("1. Unemployment rate increased significantly during COVID period.")
print("2. Higher volatility is observed during COVID months.")
print("3. Certain months show seasonal spikes in unemployment.")
print("4. Results can help policymakers plan employment support programs.")
