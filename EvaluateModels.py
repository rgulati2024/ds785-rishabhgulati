# -*- coding: utf-8 -*-
"""
Evaluate model and hyperparameter tuning
"""
import ModelBuild
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# Feature set (X) - selecting relevant engineered features
features = [
    'hour_of_day', 'day_of_week', 'is_weekend', 'run_count', 'previous_fail_flag',
    'time_since_last_run', 'failure_rate', 'avg_run_duration', 'run_duration_variance',
    'rolling_avg_run_time', 'rolling_failure_rate'
]

X = ModelBuild.df_model[features]
y = ModelBuild.df_model['fail_flag']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grids
param_grids = {
    "Logistic Regression": {
        "penalty": ['l1', 'l2'],
        "C": [0.01, 0.1, 1, 10],
        "solver": ['liblinear', 'saga']
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10]
    }
}

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Dictionary to store best model results
best_results = {}

# Train, tune, and evaluate each model
for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model from grid search
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    
    # Store results
    best_results[model_name] = {
        "Best Params": grid_search.best_params_,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc
    }
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, AUC-ROC: {auc_roc:.2f}")
    print("-" * 60)


# Correlation Heatmap of Features
plt.figure(figsize=(10, 8))
correlation_matrix = ModelBuild.df_realistic[['run_time', 'failure_rate', 'avg_run_duration', 'run_duration_variance', 'time_since_last_run']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Features")
plt.show()


