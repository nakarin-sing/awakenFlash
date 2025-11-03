# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark_final.py
FINAL CI-SAFE VERSION
- VotingClassifier bug fix (voting='soft')
- Memory + time profiling
- Poly2Wrapper C=0.1
- FIX: Ensure full DataFrame output in log
"""

import numpy as np
import pandas as pd
import time
import tracemalloc

# =====================
# matplotlib safe import
# =====================
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    HAVE_MPL = True
except ModuleNotFoundError:
    print("matplotlib/seaborn not found, skipping plots")
    HAVE_MPL = False

# ----------------------
# sklearn + xgboost (ส่วนเดิม)
# ----------------------
from sklearn.datasets import load_breast_cancer
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
# Wrappers (ส่วนเดิม)
# ----------------------
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, degree=2, C=0.1):
        self.degree = degree
        self.C = C
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.clf = LogisticRegression(C=self.C, max_iter=5000, random_state=42)
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
        self.clf = LogisticRegression(C=self.C, max_iter=5000, random_state=42)
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
# Dataset (ส่วนเดิม)
# ----------------------
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------
# Pipelines & Ensemble (ส่วน voting='soft' ที่แก้ไขแล้ว)
# ----------------------
pipe_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb.XGBClassifier(max_depth=3, n_estimators=200,
                              use_label_encoder=False, eval_metric='logloss', random_state=42))
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
    voting='soft' # ใช้ soft voting เพื่อแก้ปัญหา Ensemble
)

models = {"XGBoost": pipe_xgb, "Poly2": pipe_poly2, "RFF": pipe_rff, "Ensemble": ensemble}

# ----------------------
# Benchmark (with time + memory)
# ----------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    try:
        # CV safe: skip for Ensemble
        if name != "Ensemble":
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            cv_mean_acc = cv_scores.mean()
        else:
            cv_mean_acc = np.nan

        # Start memory & time tracking
        tracemalloc.start()
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mem_current, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Predict
        t0 = time.time()
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        predict_time = time.time() - t0

        # Save results
        results.append({
            'Model': name,
            'Train ACC': accuracy_score(y_train, y_train_pred),
            'Test ACC': accuracy_score(y_test, y_test_pred),
            'Train F1': f1_score(y_train, y_train_pred, average='weighted'),
            'Test F1': f1_score(y_test, y_test_pred, average='weighted'),
            'CV mean ACC': cv_mean_acc,
            'Train time (s)': round(train_time, 4),
            'Predict time (s)': round(predict_time, 4),
            'Memory peak (MB)': round(mem_peak/1e6, 4)
        })
    except Exception as e:
        print(f"Failed {name}: {e}")

df = pd.DataFrame(results)

# ----------------------
# FIX: ส่วนที่เพิ่มเพื่อแสดงผลลัพธ์ใน Log อย่างสมบูรณ์
# ----------------------
# ตั้งค่า Pandas ให้แสดงผลลัพธ์ทั้งหมดโดยไม่ตัดทอน
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# พิมพ์ DataFrame ทั้งหมดลงใน Log โดยใช้ .to_string()
print("\n--- Benchmark Results (Full) ---\n")
print(df.to_string())
df.to_csv("benchmark_results.csv", index=False)
# ----------------------
# Optional plot (ส่วนเดิม)
# ----------------------
if HAVE_MPL:
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x='Model', y='Test ACC')
    plt.title("Benchmark Test ACC")
    plt.tight_layout()
    plt.savefig("benchmark_test_acc.png", dpi=300)
