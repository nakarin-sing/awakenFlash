#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AWAKEN vΩ.3 — REAL-WORLD BENCHMARK
"เปรียบเทียบ AWAKEN vΩ.3 vs XGBoost บน real-world datasets"
MIT © 2025 xAI Research
"""

import time
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import resource
import xgboost as xgb

# ========================================
# AWAKEN vΩ.3 Fixed One-Step
# ========================================
class FixedOneStep:
    def __init__(self):
        self.W = None

    def fit(self, X, y):
        y_onehot = np.eye(np.max(y)+1)[y]
        X_pinv = np.linalg.pinv(X)
        self.W = X_pinv @ y_onehot

    def predict(self, X):
        logits = X @ self.W
        return np.argmax(logits, axis=1)

# ========================================
# Datasets
# ========================================
datasets = {
    "BreastCancer": load_breast_cancer(),
    "Iris": load_iris(),
    "Wine": load_wine()
}

# ========================================
# Run Benchmark
# ========================================
results = []
for name, data in datasets.items():
    X, y = data.data.astype(np.float32), data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost
    model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=-1, tree_method='hist', verbosity=0)
    t0 = time.time()
    model_xgb.fit(X_train, y_train)
    xgb_time = time.time() - t0
    xgb_pred = model_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average='macro')

    # AWAKEN vΩ.3
    model_awaken = FixedOneStep()
    t0 = time.time()
    model_awaken.fit(X_train, y_train)
    awaken_time = time.time() - t0
    awaken_pred = model_awaken.predict(X_test)
    awaken_acc = accuracy_score(y_test, awaken_pred)
    awaken_f1 = f1_score(y_test, awaken_pred, average='macro')

    results.append({
        "dataset": name,
        "xgb_acc": xgb_acc, "xgb_f1": xgb_f1, "xgb_time": xgb_time,
        "awaken_acc": awaken_acc, "awaken_f1": awaken_f1, "awaken_time": awaken_time
    })

# ========================================
# Print Table
# ========================================
print("\n" + "="*80)
print("AWAKEN vΩ.3 vs XGBoost — REAL-WORLD BENCHMARK")
print("="*80)
print(f"{'Dataset':<12} {'Model':<12} {'ACC':<8} {'F1':<8} {'Time(s)':<8}")
print("-"*80)
for r in results:
    print(f"{r['dataset']:<12} {'XGBoost':<12} {r['xgb_acc']:.4f} {r['xgb_f1']:.4f} {r['xgb_time']:.3f}")
    print(f"{r['dataset']:<12} {'AWAKEN':<12} {r['awaken_acc']:.4f} {r['awaken_f1']:.4f} {r['awaken_time']:.3f}")
print("="*80)
print(f"RAM Start: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MB")
print(f"RAM End:   {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MB")
