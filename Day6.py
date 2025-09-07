import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
print("All Libraries imported Successfully!")
iris = load_iris()

X=iris.data
y=iris.target
feature_names=iris.feature_names
class_names=iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)

print(f"\nIris dataset loaded and Split  successfully with {X.shape[0]} samples and {X.shape[1]} features.")


print("\n Training a Controlled Decision tree(max_depth=3)")

tree_clf=DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X_train,y_train)
print("Decision Tree model trained on the training set.")

print("\n Visualizing the Decision Tree")
plt.figure(figsize=(20,10))

plot_tree(tree_clf, filled=True,feature_names=feature_names,class_names=class_names,rounded=True,fontsize=12)

plt.title("Decision Tree for Iris Classification (max_deth=3),fontsize=16")
plt.show()

importances= tree_clf.feature_importances_

feature_importance_df=pd.DataFrame({'feature':feature_names,'importances':importances,})

feature_importance_df=feature_importance_df.sort_values('importances',ascending=False)


plt.figure(figsize=(10,6))
sns.barplot(x='importances',y='feature',data=feature_importance_df,palette='viridis')

plt.title("Feature Importances in the Decision Tree",fontsize=16)
plt.xlabel("Importance",fontsize=14)
plt.ylabel("Feature",fontsize=14)
plt.show()

print("\n Feature importances:")
print(feature_importance_df)

print("\n Optimizing With Cost_Complexity pruning ")

full_tree=DecisionTreeClassifier(random_state=42)
path=full_tree.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas=path.ccp_alphas

clfs=[]
for ccp_alpha in ccp_alphas:
    clf=DecisionTreeClassifier(random_state=42,ccp_alpha=ccp_alpha)
    clf.fit(X_train,y_train)
    clfs.append(clf)

clfs=clfs[:-1]
ccp_alphas=ccp_alphas[:-1]

test_scores=[clf.score(X_test,y_test) for clf in clfs]
best_alpha=ccp_alphas[np.argmax(test_scores)]

print(f"\n Best alpha found: {best_alpha:.4f}")


purned_tree=DecisionTreeClassifier(random_state=42,ccp_alpha=best_alpha)
purned_tree.fit(X_train,y_train)

print("Final Pruned Decision Tree model trained Successfully.")

plt.figure(figsize=(20,10))
plot_tree(purned_tree,filled=True,feature_names=feature_names,class_names=class_names,rounded=True,fontsize=12)

plt.title("Optimally Pruned Decision Tree for Iris Classification",fontsize=16 )
plt.show()
