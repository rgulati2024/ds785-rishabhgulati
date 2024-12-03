# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:05:00 2024

@author: GULO1L
"""
import DataPrep
import Chart
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Train models again for feature importance and evaluation plots
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}


# Drop rows where critical features are missing, if any
df_model = DataPrep.df_realistic.dropna()

# Feature set (X) - selecting relevant engineered features
features = [
    'hour_of_day', 'day_of_week', 'is_weekend', 'run_count', 'previous_fail_flag',
    'time_since_last_run', 'failure_rate', 'avg_run_duration', 'run_duration_variance',
    'rolling_avg_run_time', 'rolling_failure_rate'
]

# Dictionary to store model predictions and probabilities
y_preds = {}
y_probs = {}

X = df_model[features]
y = df_model['fail_flag']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train each model and store predictions
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_preds[model_name] = model.predict(X_test_scaled)
    y_probs[model_name] = model.predict_proba(X_test_scaled)[:, 1]  # probability of positive class

# Plot 1: Feature Importances for Tree-Based Models
for model_name in ["Random Forest", "Gradient Boosting"]:
    if model_name in models:
        model = models[model_name]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importance - {model_name}")
        sns.barplot(x=[features[i] for i in indices], y=importances[indices])
        plt.xticks(rotation=90)
        plt.ylabel("Importance")
        plt.show()

# Plot 2: Confusion Matrix for Each Model
for model_name, y_pred in y_preds.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Plot 3: ROC Curve for Each Model
plt.figure(figsize=(10, 6))
for model_name, y_prob in y_probs.items():
    fpr, tpr, _ = metrics.roc_curve(y_test, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Plot 4: Precision-Recall Curve for Each Model
plt.figure(figsize=(10, 6))
for model_name, y_prob in y_probs.items():
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=model_name)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.show()
