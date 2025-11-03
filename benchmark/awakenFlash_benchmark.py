# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb

# ==========================
# Poly2 Wrapper (ClassifierReady)
# ==========================
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.model = LogisticRegression(C=self.C, max_iter=5000)

    def fit(self, X, y):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# ==========================
# RFF Wrapper (Random Fourier Features)
# ==========================
class RFFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=100, gamma=1.0):
        self.n_components = n_components
        self.gamma = gamma
        self.model = LogisticRegression(max_iter=5000)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.W = np.random.normal(scale=np.sqrt(2 * self.gamma), size=(n_features, self.n_components))
        self.b = np.random.uniform(0, 2*np.pi, size=self.n_components)
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        self.model.fit(Z, y)
        return self

    def predict(self, X):
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        return self.model.predict(Z)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# ==========================
# Load dataset
# ==========================
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.63, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# Models
# ==========================
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric='logloss'
)
poly_model = Poly2Wrapper(C=1.0)
rff_model = RFFWrapper(n_components=300, gamma=0.1)

# ==========================
# Voting Ensemble
# ==========================
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('poly', poly_model),
        ('rff', rff_model)
    ],
    voting='soft'
)

# ==========================
# Train & evaluate function
# ==========================
def evaluate_model(model, name):
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    try:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cv_mean = cv_scores.mean()
    except:
        cv_mean = np.nan
    print(f"{name:<10} | Train ACC={train_acc:.4f}, Test ACC={test_acc:.4f}, F1={f1:.4f}, CV mean={cv_mean:.4f}")

# ==========================
# Run benchmark
# ==========================
for name, model in [('XGBoost', xgb_model), ('Poly2', poly_model), ('RFF', rff_model), ('Ensemble', ensemble)]:
    evaluate_model(model, name)
