# awakenFlash_benchmark_fixed.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

# -----------------------------
# ตัวอย่าง wrapper ของคุณ
# -----------------------------
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import PolynomialFeatures

class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, degree=2, C=1.0):
        self.degree = degree
        self.C = C
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.clf = LogisticRegression(C=self.C, max_iter=5000)
    
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.clf.fit(X_poly, y)
        return self
    
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.clf.predict(X_poly)
    
    def predict_proba(self, X):
        X_poly = self.poly.transform(X)
        return self.clf.predict_proba(X_poly)
    
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
        X_rff = self.rff.fit_transform(X)
        self.clf.fit(X_rff, y)
        return self
    
    def predict(self, X):
        X_rff = self.rff.transform(X)
        return self.clf.predict(X_rff)
    
    def predict_proba(self, X):
        X_rff = self.rff.transform(X)
        return self.clf.predict_proba(X_rff)
    
    @property
    def _estimator_type(self):
        return "classifier"

# -----------------------------
# Load your real dataset here
# -----------------------------
# สำหรับตัวอย่าง ผมใช้ sklearn breast_cancer dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

# -----------------------------
# Train/Test split (stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# สร้าง pipeline สำหรับแต่ละ model
# -----------------------------
pipe_xgb = make_pipeline(
    StandardScaler(),
    xgb.XGBClassifier(
        max_depth=3,
        n_estimators=200,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
)
pipe_poly2 = make_pipeline(StandardScaler(), Poly2Wrapper(degree=2, C=0.1))
pipe_rff = make_pipeline(StandardScaler(), RFFWrapper(gamma='scale', n_components=100, C=1.0))

# -----------------------------
# Ensemble
# -----------------------------
ensemble = VotingClassifier(
    estimators=[
        ('XGBoost', pipe_xgb),
        ('Poly2', pipe_poly2),
        ('RFF', pipe_rff)
    ],
    voting='hard'
)

# -----------------------------
# Cross-validation
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "XGBoost": pipe_xgb,
    "Poly2": pipe_poly2,
    "RFF": pipe_rff,
    "Ensemble": ensemble
}

results = []

for name, model in models.items():
    # CV
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    
    # Fit on training set
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'Train ACC': accuracy_score(y_train, y_train_pred),
        'Test ACC': accuracy_score(y_test, y_test_pred),
        'Train F1': f1_score(y_train, y_train_pred, average='weighted'),
        'Test F1': f1_score(y_test, y_test_pred, average='weighted'),
        'CV mean ACC': cv_mean
    })

df_results = pd.DataFrame(results)
print(df_results)
