
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


titanic_df=sns.load_dataset('titanic')

print("Printing Missing Values before Cleaning:")
print(titanic_df.isnull().sum())

original_age=titanic_df['age'].copy()

titanic_df['age']=titanic_df.groupby('pclass')['age'].transform(lambda x:x.fillna(x.median()))

print("/n Successfully emputed missing values in 'age' column using median age based on passenger class.")

embarked_mode=titanic_df['embarked'].mode()[0]

titanic_df['embarked'].fillna(embarked_mode,inplace=True)
print("Successfully imputed missing values in 'embarked' column using mode.")

titanic_df.drop('deck',axis=1,inplace=True)
print("Dropped 'deck' column due to high number of missing values.")

titanic_df.drop('embark_town',axis=1,inplace=True)
print("Dropped 'embark_town' column as it is redundant with 'embarked' column.")

print("\nPrinting Missing Values after Cleaning:")
print(titanic_df.isnull().sum())


plt.figure(figsize=(12,6))

sns.histplot(original_age,kde=True,color='blue',label='Before Imputation',bins=30)
sns.histplot(titanic_df['age'],kde=True,color='orange',label='After Imputation',bins=30)

plt.title('Distribution of Age Before and After Median Imputation by Class')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()



plt.figure(figsize=(10, 6))
sns.heatmap(titanic_df.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values in the Original Titanic Dataset')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=titanic_df)
plt.title('Distribution of Age by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()