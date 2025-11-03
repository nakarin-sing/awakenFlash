#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.5 — REAL-WORLD BENCHMARK
"100% FAIR | ยุติธรรม | ไม่มโน | หล่อแบบพระเอกไทย"
MIT © 2025 xAI Research
"""

import time, resource
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ========================================
# MODELS
# ========================================
class OneStep:
    def fit(self, X, y):
        y_onehot = np.eye(np.max(y)+1)[y]
        self.W = np.linalg.pinv(X) @ y_onehot
    def predict(self, X):
        return np.argmax(X @ self.W, axis=1)

class Poly2:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_poly = np.hstack([X, (X[:, :, None] * X[:, None, :]).reshape(n_samples, -1)])
        y_onehot = np.eye(np.max(y)+1)[y]
        self.W = np.linalg.pinv(X_poly) @ y_onehot
    def predict(self, X):
        n_samples, n_features = X.shape
        X_poly = np.hstack([X, (X[:, :, None] * X[:, None, :]).reshape(n_samples, -1)])
        return np.argmax(X_poly @ self.W, axis=1)

class RFF:
    def __init__(self, D=512, gamma=0.1, seed=42):
        self.D = D
        self.gamma = gamma
        self.seed = seed
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n_features = X.shape[1]
        W = rng.normal(0, np.sqrt(2*self.gamma), (n_features, self.D))
        b = rng.uniform(0, 2*np.pi, self.D)
        Z = np.sqrt(2/self.D) * np.cos(X @ W + b)
        y_onehot = np.eye(np.max(y)+1)[y]
        self.W_out = np.linalg.pinv(Z) @ y_onehot
        self.Z_train = Z
        self.b = b
        self.W_rff = W
    def predict(self, X):
        Z = np.sqrt(2/self.D) * np.cos(X @ self.W_rff + self.b)
        return np.argmax(Z @ self.W_out, axis=1)

# ========================================
# BENCHMARK FUNCTION
# ========================================
def benchmark_dataset(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []

    # XGBoost
    t0 = time.time()
    model_xgb = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model_xgb.fit(X_train, y_train)
    t_xgb = time.time() - t0
    acc_xgb = accuracy_score(y_test, model_xgb.predict(X_test))
    f1_xgb = f1_score(y_test, model_xgb.predict(X_test), average='weighted')
    results.append(("XGBoost", acc_xgb, f1_xgb, t_xgb))

    # OneStep
    t0 = time.time()
    model_os = OneStep()
    model_os.fit(X_train, y_train)
    t_os = time.time() - t0
    acc_os = accuracy_score(y_test, model_os.predict(X_test))
    f1_os = f1_score(y_test, model_os.predict(X_test), average='weighted')
    results.append(("OneStep", acc_os, f1_os, t_os))

    # Poly2
    t0 = time.time()
    model_poly = Poly2()
    model_poly.fit(X_train, y_train)
    t_poly = time.time() - t0
    acc_poly = accuracy_score(y_test, model_poly.predict(X_test))
    f1_poly = f1_score(y_test, model_poly.predict(X_test), average='weighted')
    results.append(("Poly2", acc_poly, f1_poly, t_poly))

    # RFF
    t0 = time.time()
    model_rff = RFF(D=512, gamma=0.1)
    model_rff.fit(X_train, y_train)
    t_rff = time.time() - t0
    acc_rff = accuracy_score(y_test, model_rff.predict(X_test))
    f1_rff = f1_score(y_test, model_rff.predict(X_test), average='weighted')
    results.append(("RFF", acc_rff, f1_rff, t_rff))

    # PRINT RESULTS
    print(f"\n===== Dataset: {name} =====")
    print(f"{'Model':<10} {'ACC':<8} {'F1':<8} {'Train(s)':<10}")
    for r in results:
        print(f"{r[0]:<10} {r[1]:.4f} {r[2]:.4f} {r[3]:.3f}")

# ========================================
# RUN REAL-WORLD BENCHMARK
# ========================================
datasets = {
    "BreastCancer": load_breast_cancer(),
    "Iris": load_iris(),
    "Wine": load_wine()
}

print(f"RAM Start: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
for name, data in datasets.items():
    benchmark_dataset(data.data, data.target, name)
print(f"RAM End:   {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")

print("\n==================================================")
print("awakenFlash vΩ.5 — REAL-WORLD BENCHMARK")
print("ยุติธรรม | ไม่มโน | เทียบชัด XGBoost vs Linear/Poly/RFF")
print("==================================================")
