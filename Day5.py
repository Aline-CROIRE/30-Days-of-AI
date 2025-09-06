import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("All Libraries imported Successfully!")

iris=load_iris()

X=pd.DataFrame(iris.data, columns=iris.feature_names)

y=pd.Series(iris.target)

print(f"\nIris dataset loaded successfully with {X.shape[0]} samples and {X.shape[1]} features.")
print("\nFirst 5 rows of features (X):")
print(X.head())
print("\nTarget variable (y) value counts:")
print(y.value_counts().rename({i: name for i, name in enumerate(iris.target_names)}))

print("\nData Preparation:")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
print(f"Data split into training set (rows:{X_train.shape[0]}) and testing set (rows:{X_test.shape[0]}).")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaling applied using StandardScaler.")

print("\n KNN Model Training and cross-validation")

knn=KNeighborsClassifier(n_neighbors=3)

X_scaled= scaler.fit_transform(X)

cv_scores=cross_val_score(knn,X_scaled,y,cv=5)

print(f"Cross-validation scores(5 folds): {cv_scores}")
print(f"Average CV score:{cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")

knn.fit(X_train_scaled,y_train)

print("\n model trained on the training set.")

print("\n Model Comparison")

tree_clf=DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_scaled,y_train)
print("Decision Tree model trained on the training set.")

y_pred_knn=knn.predict(X_test_scaled)
y_pred_tree=tree_clf.predict(X_test_scaled)

print("\n KNN Model Classification Report:")
print(classification_report(y_test,y_pred_knn,target_names=iris.target_names))

print("\n Decision Tree Model Classification Report:")
print(classification_report(y_test,y_pred_tree,target_names=iris.target_names))


def plot_decision_boundaries(X, y, model, title):

    X_vis = X[:, :2]
  
    model.fit(X_vis, y)


    x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
    y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
   
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

 
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title(title, fontsize=16)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(iris.target_names))
    plt.show()


print("\n--- Visualizing Decision Boundaries (using first two features) ---")

X_unscaled = iris.data
y_unscaled = iris.target


knn_vis = KNeighborsClassifier(n_neighbors=3)
plot_decision_boundaries(X_unscaled, y_unscaled, knn_vis, "KNN Decision Boundaries (k=3)")

tree_vis = DecisionTreeClassifier(random_state=42)
plot_decision_boundaries(X_unscaled, y_unscaled, tree_vis, "Decision Tree Decision Boundaries")


wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names) 
y = pd.Series(wine.target)                              

print(f"\nDataset loaded. X shape: {X.shape}")
print("First 5 rows of features (X):")
print(X.head())
print("\nTarget wine classes (y) value counts:")

print(y.value_counts().rename({i: name for i, name in enumerate(wine.target_names)}))

print("\n--- Data Preparation ---")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Data split. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler.")

print("\n--- KNN Model Training and Cross-Validation ---")


knn = KNeighborsClassifier(n_neighbors=3)


X_scaled = scaler.fit_transform(X) 
cv_scores = cross_val_score(knn, X_scaled, y, cv=5)

print(f"Cross-Validation Scores (5 folds): {cv_scores}")
print(f"Average CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


knn.fit(X_train_scaled, y_train)
print("\nKNN model trained on the training set.")

print("\n--- Model Comparison ---")


tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train_scaled, y_train)
print("Decision Tree model trained.")


y_pred_knn = knn.predict(X_test_scaled)
y_pred_tree = tree_clf.predict(X_test_scaled)

print("\n--- K-Nearest Neighbors (KNN) Classification Report ---")
print(classification_report(y_test, y_pred_knn, target_names=wine.target_names))

print("\n--- Decision Tree Classification Report ---")
print(classification_report(y_test, y_pred_tree, target_names=wine.target_names))

def plot_decision_boundaries(X, y, model, title, feature_names, target_names):
    X_vis = X[:, :2] 
    model.fit(X_vis, y)

    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title, fontsize=16)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(target_names))
    plt.show()

print("\n--- Visualizing Decision Boundaries (using 'alcohol' and 'malic_acid') ---")

X_unscaled = wine.data
y_unscaled = wine.target


knn_vis = KNeighborsClassifier(n_neighbors=3)
plot_decision_boundaries(X_unscaled, y_unscaled, knn_vis, "KNN Decision Boundaries (k=3)", wine.feature_names, wine.target_names)


tree_vis = DecisionTreeClassifier(random_state=42)
plot_decision_boundaries(X_unscaled, y_unscaled, tree_vis, "Decision Tree Decision Boundaries", wine.feature_names, wine.target_names)