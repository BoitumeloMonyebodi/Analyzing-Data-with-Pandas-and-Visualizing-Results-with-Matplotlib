import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load and Explore the Dataset
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    df['species'] = df['target'].map(dict(enumerate(iris_data.target_names)))

    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nMissing values in each column:")
    print(df.isnull().sum())

    # Clean dataset (no missing values in this case)
except Exception as e:
    print("An error occurred while loading the dataset:", e)

# Basic Data Analysis
print("\nStatistical Summary:")
print(df.describe())

# Group by species and compute mean
grouped = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped)

# Task 3: Data Visualization
sns.set(style="whitegrid")

# 1. Line Chart (simulated trend using index)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title('Line Chart of Sepal and Petal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.tight_layout()
plt.savefig("line_chart.png")
plt.show()

# 2. Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

# 3. Histogram
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histogram of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# 4. Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()
