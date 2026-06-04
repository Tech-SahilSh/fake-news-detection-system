# train.py
import pandas as pd
import numpy as np
import nltk
import string
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    stop_words = stopwords.words('english')
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return ' '.join([word for word in text.split() if word not in stop_words])

# Load and label datasets
fake = pd.read_csv("data/Fake.csv")
real = pd.read_csv("data/Real.csv")
fake["label"] = 0
real["label"] = 1
df = pd.concat([fake[['text', 'label']], real[['text', 'label']]])
df.dropna(inplace=True)

# Preprocess text
df['text'] = df['text'].apply(preprocess_text)

# Train-test split
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# K-Fold Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === MODEL DEFINITIONS & TUNING ===

# 1. Logistic Regression
log_reg = LogisticRegression(max_iter=500)
log_params = {'C': [0.01, 0.1, 1, 10]}
log_grid = GridSearchCV(log_reg, log_params, cv=cv, scoring='f1', n_jobs=-1)
log_grid.fit(X_train_vec, y_train)

# 2. Linear Regression (as classifier)
lin_reg = LinearRegression()
lin_reg.fit(X_train_vec, y_train)
lin_preds = np.where(lin_reg.predict(X_test_vec) > 0.5, 1, 0)

# 3. Random Forest
rf = RandomForestClassifier()
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
rf_search = RandomizedSearchCV(rf, rf_params, n_iter=5, cv=cv, scoring='f1', n_jobs=-1, random_state=42)
rf_search.fit(X_train_vec, y_train)

# 4. Gradient Boosting
gb = GradientBoostingClassifier(learning_rate=0.5)
gb_params = {
    'n_estimators': [100, 150],
    'max_depth': [3, 5],
    'min_samples_split': [2, 4],
}
gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring='f1', n_jobs=-1)
gb_grid.fit(X_train_vec, y_train)

# === EVALUATION ===

def evaluate_model(name, model, X_test, y_test):
    if name == "Linear Regression":
        y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)
    else:
        y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

results = {
    "Logistic Regression": evaluate_model("Logistic Regression", log_grid.best_estimator_, X_test_vec, y_test),
    "Linear Regression": evaluate_model("Linear Regression", lin_reg, X_test_vec, y_test),
    "Random Forest": evaluate_model("Random Forest", rf_search.best_estimator_, X_test_vec, y_test),
    "Gradient Boosting": evaluate_model("Gradient Boosting", gb_grid.best_estimator_, X_test_vec, y_test)
}

# Save everything
with open("models/all_models.pkl", "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "models": {
            "Logistic Regression": log_grid.best_estimator_,
            "Linear Regression": lin_reg,
            "Random Forest": rf_search.best_estimator_,
            "Gradient Boosting": gb_grid.best_estimator_
        },
        "metrics": results
    }, f)

# Print results
for model, scores in results.items():
    print(f"\n{model} Evaluation:")
    for metric, score in scores.items():
        print(f"{metric.capitalize()}: {score:.4f}")


