import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. Data Loading and Preparation ---
print("--- Loading and Preparing Data ---")
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
print("Dataset loaded successfully.")
print(df.head())


# --- 2. Feature Extraction (TF-IDF) and Data Splitting ---
print("\n--- Splitting Data and Performing TF-IDF Vectorization ---")
X = df['message']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"Data split and vectorized. Training data shape: {X_train_tfidf.shape}")


# --- 3. Model Training and Hyperparameter Tuning ---
print("\n--- Training and Tuning the Multinomial Naive Bayes Model ---")
nb_classifier = MultinomialNB()
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

grid_search = GridSearchCV(nb_classifier, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

best_nb_model = grid_search.best_estimator_

print(f"Best alpha found by GridSearchCV: {grid_search.best_params_['alpha']}")


# --- 4. Model Evaluation ---
print("\n--- Evaluating the Best Model ---")
y_pred = best_nb_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy:.4f}")
print(f"F1-Score on Test Set: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


# --- 5. ROC Curve Visualization ---
print("\n--- Generating ROC Curve ---")
y_pred_proba = best_nb_model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# --- 6. Testing on New, Unseen Messages ---
print("\n--- Testing on New, Unseen Messages ---")
new_messages = [
    "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/12345 to claim now.",
    "Hey, are we still on for dinner tonight at 7pm?",
    "URGENT: Your account has been compromised. Please click this link to secure your account immediately.",
    "Thanks for setting up the meeting. I'll send over the notes shortly.",
    "WINNER!! As a valued customer, you have been selected to receive a free cruise to the Bahamas. Reply to claim."
]

new_messages_tfidf = tfidf_vectorizer.transform(new_messages)
predictions = best_nb_model.predict(new_messages_tfidf)
prediction_labels = ['ham' if p == 0 else 'spam' for p in predictions]

for msg, label in zip(new_messages, prediction_labels):
    print(f'Message: "{msg}"\nPredicted: {label.upper()}\n')
