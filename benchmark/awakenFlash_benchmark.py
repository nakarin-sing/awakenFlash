# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark_stable_v3.py
==================================
- Fully stable: GitHub Actions safe
- VotingClassifier + Pipeline + Custom Wrappers
- Stratified train/test split + CV
- Handles multi-class datasets
- Export CSV/LaTeX + plot Test ACC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------
# Custom Wrappers
# ----------------------
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, degree=2, C=1.0):
        self.degree = degree
        self.C = C
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.clf = LogisticRegression(C=self.C, max_iter=5000)
    def fit(self, X, y):
        self.clf.fit(self.poly.fit_transform(X), y)
        return self
    def predict(self, X):
        return self.clf.predict(self.poly.transform(X))
    def predict_proba(self, X):
        return self.clf.predict_proba(self.poly.transform(X))
    @property
    def _estimator_type(self):
        return "classifier"

class RFFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma='scale', n_components=100, C=1.0):
        self.gamma = gamma
        self.n_components = n_components
        self.C = C
        self.rff = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=42)
        self.clf = LogisticRegression(C=self.C, max_iter=5000)
    def fit(self, X, y):
        self.clf.fit(self.rff.fit_transform(X), y)
        return self
    def predict(self, X):
        return self.clf.predict(self.rff.transform(X))
    def predict_proba(self, X):
        return self.clf.predict_proba(self.rff.transform(X))
    @property
    def _estimator_type(self):
        return "classifier"

# ----------------------
# Datasets
# ----------------------
datasets = {
    "BreastCancer": load_breast_cancer(),
    "Wine": load_wine(),
    "Iris": load_iris()
}

# ----------------------
# Models & Pipelines
# ----------------------
def make_models():
    pipe_xgb = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            max_depth=3, n_estimators=200, use_label_encoder=False,
            eval_metric='logloss', random_state=42
        ))
    ])
    pipe_poly2 = Pipeline([
        ('scaler', StandardScaler()),
        ('poly2', Poly2Wrapper(degree=2, C=0.1))
    ])
    pipe_rff = Pipeline([
        ('scaler', StandardScaler()),
        ('rff', RFFWrapper(gamma='scale', n_components=100, C=1.0))
    ])
    ensemble = VotingClassifier(
        estimators=[
            ('XGBoost', pipe_xgb),
            ('Poly2', pipe_poly2),
            ('RFF', pipe_rff)
        ],
        voting='hard'
    )
    return {
        "XGBoost": pipe_xgb,
        "Poly2": pipe_poly2,
        "RFF": pipe_rff,
        "Ensemble": ensemble
    }

# ----------------------
# Benchmark runner
# ----------------------
results = []

for dname, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    models = make_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            results.append({
                'Dataset': dname,
                'Model': name,
                'Train ACC': accuracy_score(y_train, y_train_pred),
                'Test ACC': accuracy_score(y_test, y_test_pred),
                'Train F1': f1_score(y_train, y_train_pred, average='weighted'),
                'Test F1': f1_score(y_test, y_test_pred, average='weighted'),
                'CV mean ACC': cv_scores.mean()
            })
        except Exception as e:
            print(f"Failed {name} on {dname}: {e}")

df = pd.DataFrame(results)
print(df)

# ----------------------
# Export results
# ----------------------
df.to_csv("benchmark_results.csv", index=False)
with open("benchmark_results.tex", "w") as f:
    f.write(df.to_latex(index=False))

# ----------------------
# Plot Test Accuracy
# ----------------------
plt.figure(figsize=(12,6))
sns.barplot(data=df, x='Dataset', y='Test ACC', hue='Model')
plt.title("Benchmark Test Accuracy")
plt.ylabel("Test Accuracy")
plt.tight_layout()
plt.savefig("benchmark_test_acc.png", dpi=300)
plt.show()
