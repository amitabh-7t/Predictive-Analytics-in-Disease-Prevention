import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
data = pd.read_csv('cardio_train.csv', sep=';')

# Overview of the data
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Since there are no missing values shown in the sample, we skip dropna or fillna.

# Remove duplicates if any
data = data.drop_duplicates()

# Histograms for numerical columns
data['age'].hist(bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age (in days)')
plt.ylabel('Frequency')
plt.show()

data['height'].hist(bins=30, edgecolor='black')
plt.title('Height Distribution')
plt.xlabel('Height (in cm)')
plt.ylabel('Frequency')
plt.show()

data['weight'].hist(bins=30, edgecolor='black')
plt.title('Weight Distribution')
plt.xlabel('Weight (in kg)')
plt.ylabel('Frequency')
plt.show()

# Convert age from days to years for better interpretation
data['age_years'] = data['age'] // 365

# Boxplots to check for outliers in numerical columns
sns.boxplot(x=data['height'])
plt.title('Height Boxplot')
plt.show()

sns.boxplot(x=data['weight'])
plt.title('Weight Boxplot')
plt.show()

# Countplot for categorical variables (gender, cholesterol, smoke, alco, active)
sns.countplot(x='gender', data=data)
plt.title('Gender Distribution')
plt.show()

sns.countplot(x='cholesterol', data=data)
plt.title('Cholesterol Levels')
plt.show()

sns.countplot(x='smoke', data=data)
plt.title('Smoking Habit')
plt.show()

sns.countplot(x='alco', data=data)
plt.title('Alcohol Consumption')
plt.show()

sns.countplot(x='active', data=data)
plt.title('Physical Activity')
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot to visualize relationships between variables
sns.pairplot(data[['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cardio']])
plt.title('Pairplot of Selected Features')
plt.show()

# Boxplot of categorical vs. numerical data
sns.boxplot(x='cholesterol', y='weight', data=data)
plt.title('Weight vs. Cholesterol Levels')
plt.show()

# Violin plot for more detailed comparison
sns.violinplot(x='cholesterol', y='weight', data=data)
plt.title('Weight vs. Cholesterol Levels')
plt.show()

# Identifying Outliers using Z-scores
z_scores = np.abs(stats.zscore(data[['height', 'weight', 'ap_hi', 'ap_lo']]))
outliers = data[(z_scores > 3).any(axis=1)]
print(f'Number of Outliers: {len(outliers)}')
print(outliers)

# If you want to remove the outliers:
data_cleaned = data[(z_scores < 3).all(axis=1)]
print(f'Cleaned Data Shape: {data_cleaned.shape}')
