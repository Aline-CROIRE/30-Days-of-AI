
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic_df=sns.load_dataset('titanic')

# Display the first few rows of the dataset
print("First 5 rows of the Titanic dataset:")
print(titanic_df.head())

# Display the last few rows of the dataset
print("\nlast 5 rows of the Titanic dataset:")
print(titanic_df.tail())

# Display basic information about the dataset
print("\nBasic information about the Titanic dataset:")
print(titanic_df.info())

# Display summary statistics of the dataset
print("\nSummary statistics of the Titanic dataset:")
print(titanic_df.describe())

#Check Duplicate Rows
duplicate_df_rows=titanic_df.duplicated().sum()
print(f"\nNumber of duplicate rows in the Titanic dataset: {duplicate_df_rows}")

#Check Missing Values
missing_values=titanic_df.isnull().sum()
print("\nMissing values in each column of the Titanic dataset:")
print(missing_values)

#Quantiles and Outliers Detection for 'fare' column

Q1=titanic_df['fare'].quantile(0.25)
Q3=titanic_df['fare'].quantile(0.75)
IQR=Q3-Q1

lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR

outliers=titanic_df[(titanic_df['fare']<lower_bound)|(titanic_df['fare']>upper_bound)]

# Display the number of outliers detected

print(f"\nNumber of outliers in the 'fare' column: {outliers.shape[0]}")


# Correlation Heatmap
corretion_matrix=titanic_df.corr(numeric_only=True)

plt.figure(figsize=(10,6))

sns.heatmap(corretion_matrix,annot=True,cmap='BrBG',fmt=".2f")

# Add title and show the plot
plt.title('Correlation Heatmap of the Titanic Dataset')
plt.show()