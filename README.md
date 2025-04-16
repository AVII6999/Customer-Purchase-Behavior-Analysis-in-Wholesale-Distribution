# Customer-Purchase-Behavior-Analysis-in-Wholesale-Distribution
1)Identify key customer segments based on purchasing behavior.  2)Determine correlation between different product categories.  3)Analyze spending trends across Channels and Regions.  4)Detect outliers and anomalies in spending.  5)Assess variance and skewness in product spending.



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew
df = pd.read_csv('Wholesale customers data.csv')
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
print(df.head())
print(df.info())
df['TotalSpend'] = df[features].sum(axis=1)
df['SpendingSegment'] = pd.qcut(df['TotalSpend'], q=3, labels=['Low', 'Medium', 'High'])

print("\nCustomer Segment Counts:\n", df['SpendingSegment'].value_counts())

sns.boxplot(x='SpendingSegment', y='TotalSpend', data=df)
plt.title("Customer Segments Based on Total Spending")
plt.show()
corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
grouped = df.groupby(['Channel', 'Region'])[features].mean().round(0)
print("\nAverage Spending by Channel & Region:\n", grouped)

grouped.plot(kind='bar', figsize=(12, 6))
plt.title("Average Spending by Channel and Region")
plt.ylabel("Mean Spend")
plt.xticks(rotation=45)
plt.legend(title="Product Category")
plt.tight_layout()
plt.show()
plt.figure(figsize=(14, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[feature])
    plt.title(feature)
plt.tight_layout()
plt.show()
Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[features] < (Q1 - 1.5 * IQR)) | (df[features] > (Q3 + 1.5 * IQR)))
print("\nOutliers per feature:\n", outliers.sum())
print("\nFeature Variance:\n", df[features].var())
print("\nFeature Skewness:\n", df[features].apply(skew))
