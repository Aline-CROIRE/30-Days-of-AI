import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score, confusion_matrix

print("All Libraries imported Succesfully!")


titanic_df = sns.load_dataset('titanic')

titanic_df['age']=titanic_df.groupby('pclass')['age'].transform(lambda x: x.fillna(x.median()))
embarked_mode=titanic_df['embarked'].mode()[0]

titanic_df['embarked'].fillna(embarked_mode,inplace=True)

titanic_df.drop(['deck','embark_town'],axis=1,inplace=True)

print('Data loaded and Cleaned Succesfully!')


titanic_df['family_size']=titanic_df['sibsp']+titanic_df['parch']+1

print("'family_size' feature created.")


titanic_df=pd.get_dummies(titanic_df,columns=['sex','embarked'],drop_first=True)
print("Categorical variables encoded using One-Hot Encoding.")


features=['pclass','age','family_size','sex_male','embarked_Q','embarked_S']
X=titanic_df[features]
y=titanic_df['survived']

print(f"Features selected for modeling: {X.columns.tolist()}")


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(f"Data split into training set (rows:{(X_train.shape[0])} and testing set (rows:{(X_test.shape[0])}.")


log_reg=LogisticRegression(solver='liblinear',max_iter=1000)

param_grid={'C':[0.001,0.01,0.1,1,10,100]}

grid_search=GridSearchCV(log_reg,param_grid,cv=5,scoring='accuracy')

grid_search.fit(X_train,y_train)


best_model=grid_search.best_estimator_

print(f"Best hyperparameters found: {grid_search.best_params_}")


y_pred=best_model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)

print("\n Model Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()




print("All Libraries imported Succesfully!")

cancer=load_breast_cancer()

X=pd.DataFrame(cancer.data,columns=cancer.feature_names)
y=pd.Series(cancer.target)

print(f"Data loaded successfully with {X.shape[0]} samples and {X.shape[1]} features.")
print( 'First 5 rows of features(X):')
print(X.head())
print('\n Target variable(y) value counts:')
print(y.value_counts())

print("\n Data Processing )")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print(f"Data split into training set (rows:{(X_train.shape[0])} and testing set (rows:{(X_test.shape[0])}.")

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.transform(X_test)
print("Feature scaling applied using StandardScaler.")



print("\n Model Training and Hyperparameter Tuning")

log_reg=LogisticRegression(random_state=42,solver='liblinear')

param_grid={'C':[0.001,0.01,0.1,1,10,100]}

grid_search =GridSearchCV(estimator=log_reg,param_grid=param_grid,cv=5,scoring='accuracy',n_jobs=-1)

grid_search.fit(x_train_scaled,y_train)

best_model=grid_search.best_estimator_

print(f"Best hyperparameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

print("\n Model Evaluation Results")

y_pred=best_model.predict(x_test_scaled)
y_prob=best_model.predict_proba(x_test_scaled)[:,1]

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)    
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(7,6))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap='Greens',xticklabels=['Predicted Maligant(0)','Predicted Benign(1)'],yticklabels=['Actual Maligant(0)','Actual Benign(1)'])

plt.title('Confusion Matrix for Breast Cancer Classification',fontsize=16)

plt.xlabel('Predicted Label',fontsize=12)
plt.ylabel('True Label',fontsize=12)
plt.show()

