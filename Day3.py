import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Data Preparation ---
titanic_df = sns.load_dataset('titanic')

titanic_df['age'] = titanic_df.groupby('pclass')['age'].transform(lambda x: x.fillna(x.median()))
embarked_mode = titanic_df['embarked'].mode()[0]
titanic_df['embarked'].fillna(embarked_mode, inplace=True)
titanic_df.drop(['deck', 'embark_town'], axis=1, inplace=True)

titanic_df['Survival Status'] = titanic_df['survived'].map({0: 'Did not Survive', 1: 'Survived'})
titanic_df['family_size'] = titanic_df['sibsp'] + titanic_df['parch'] + 1


# --- Visualization 1: Countplot for Survival by Gender ---
plt.figure(figsize=(8, 6))
sns.countplot(x='survived', hue='sex', data=titanic_df, palette='viridis')
plt.title('Survival Count by Gender', fontsize=16)
plt.xlabel('Survival Status (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.xticks([0, 1], ['Did not Survive', 'Survived'])
plt.legend(title='Gender')
plt.show()


# --- Visualization 2: Boxplot for Age by Class and Survival ---
plt.figure(figsize=(12, 8))
sns.boxplot(x='pclass', y='age', hue='Survival Status', data=titanic_df, palette='coolwarm')
plt.title('Age Distribution by Passenger Class and Survival Status', fontsize=16)
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.legend(title='Survival Status')
plt.show()


# --- Visualization 3: Pairplot of Key Features ---
pairplot_df = titanic_df[['survived', 'pclass', 'age', 'fare']]
g_pair = sns.pairplot(pairplot_df, hue='survived', palette='husl', markers=["o", "s"])
g_pair.fig.suptitle('Pairplot of Key Features by Survival Status', y=1.03, fontsize=16)
plt.tight_layout()
plt.show()


# --- Visualization 4: Countplot for Survival by Family Size ---
plt.figure(figsize=(12, 7))
sns.countplot(x='family_size', hue='Survival Status', data=titanic_df, palette='magma')
plt.title('Survival Count by Family Size', fontsize=16)
plt.xlabel('Number of Family Members Aboard', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend(title='Survival Status')
plt.show()


# --- Visualization 5: Catplot for Survival by Port and Class ---
g_cat = sns.catplot(x='pclass', hue='Survival Status', col='embarked', data=titanic_df, kind='count', palette='pastel')
g_cat.fig.suptitle('Survival in Each Passenger Class by Port of Embarkation', y=1.03, fontsize=16)
plt.tight_layout()
plt.show()


# --- Visualization 6: Violinplot for Fare by Survival ---
plt.figure(figsize=(10, 7))
sns.violinplot(x='Survival Status', y='fare', data=titanic_df, palette='muted')
plt.title('Fare Distribution by Survival Status', fontsize=16)
plt.xlabel('Survival Status', fontsize=12)
plt.ylabel('Fare Paid', fontsize=12)
plt.axhline(titanic_df['fare'].median(), color='red', linestyle='--', label=f"Median Fare: ${titanic_df['fare'].median():.2f}")
plt.legend()
plt.show()

