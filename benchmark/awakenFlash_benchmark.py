#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURE ONESTEP vs XGBOOST - 100,000 SAMPLES
- แก้ IndexError
- ใช้ classes จากโมเดลแรก
- รองรับ 3+ classes
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import psutil
import gc
from datetime import datetime

def cpu_time():
    p = psutil.Process(os.getpid())
    return p.cpu_times().user + p.cpu_times().system

# ========================================
# 1. OneStep + Nyström
# ========================================
class OneStepNystrom:
    def __init__(self, C=1.0, n_components=3000, gamma=0.05, random_state=42):
        self.C = C
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.landmarks_ = None
        self.beta_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = self.scaler.fit_transform(X).astype(np.float32)
        n, d = X.shape
        m = min(self.n_components, n)
        
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)[:m]
        self.landmarks_ = X[idx]
        
        diff = X[:, None, :] - self.landmarks_[None, :, :]
        K_nm = np.exp(-self.gamma * np.sum(diff**2, axis=2))
        
        diff_mm = self.landmarks_[:, None, :] - self.landmarks_[None, :, :]
        K_mm = np.exp(-self.gamma * np.sum(diff_mm**2, axis=2))
        
        lambda_reg = self.C * np.trace(K_mm) / m
        K_reg = K_mm + lambda_reg * np.eye(m, dtype=np.float32)
        
        self.classes_ = np.unique(y)
        y_onehot = np.zeros((n, len(self.classes_)), dtype=np.float32)
        for i, c in enumerate(self.classes_):
            y_onehot[y == c, i] = 1.0
        
        self.beta_ = np.linalg.solve(K_reg, K_nm.T @ y_onehot)
        return self

    def predict(self, X):
        X = self.scaler.transform(X).astype(np.float32)
        diff = X[:, None, :] - self.landmarks_[None, :, :]
        K_test = np.exp(-self.gamma * np.sum(diff**2, axis=2))
        scores = K_test @ self.beta_
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        X = self.scaler.transform(X).astype(np.float32)
        diff = X[:, None, :] - self.landmarks_[None, :, :]
        K_test = np.exp(-self.gamma * np.sum(diff**2, axis=2))
        scores = K_test @ self.beta_
        proba = np.exp(scores)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

# ========================================
# 2. Mini-Batch OneStep (FIXED)
# ========================================
class MiniBatchOneStep:
    def __init__(self, batch_size=10000, n_components=3000, C=1.0):
        self.batch_size = batch_size
        self.n_components = n_components
        self.C = C
        self.models = []

    def fit(self, X, y):
        n = X.shape[0]
        print(f"Mini-Batch: Training {n//self.batch_size} batches...")
        for i in range(0, n, self.batch_size):
            Xb = X[i:i+self.batch_size]
            yb = y[i:i+self.batch_size]
            model = OneStepNystrom(C=self.C, n_components=self.n_components, random_state=42+i)
            model.fit(Xb, yb)
            self.models.append(model)
        return self

    def predict(self, X):
        if not self.models:
            return np.array([])
        classes = self.models[0].classes_
        n_classes = len(classes)
        preds = np.zeros((X.shape[0], n_classes))
        
        for model in self.models:
            proba = model.predict_proba(X)
            # จัด alignment ตาม classes
            for i, c in enumerate(classes):
                if c in model.classes_:
                    idx = np.where(model.classes_ == c)[0][0]
                    preds[:, i] += proba[:, idx]
                # else: 0 (ไม่พบใน batch นี้)
        
        return classes[np.argmax(preds, axis=1)]

# ========================================
# 3. XGBoost
# ========================================
class XGBoostModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            n_jobs=1, random_state=42, tree_method='hist', verbosity=0
        )

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)

# ========================================
# 4. Save
# ========================================
def save(content):
    os.makedirs('benchmark_results', exist_ok=True)
    with open('benchmark_results/pure_vs_xgb_100k.txt', 'w') as f:
        f.write(f"# {datetime.now()}\n\n{content}\n")
    print("Saved: benchmark_results/pure_vs_xgb_100k.txt")

# ========================================
# 5. Main
# ========================================
def main():
    print("="*80)
    print("PURE ONESTEP vs XGBOOST - 100,000 SAMPLES")
    print("="*80)

    X, y = make_classification(
        n_samples=120000, n_features=20, n_informative=15,
        n_classes=3, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=20000, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    reps = 5

    # --- PURE ONESTEP ---
    print("\nTraining PURE ONESTEP (Mini-Batch)...")
    start = cpu_time()
    model_pure = MiniBatchOneStep(batch_size=10000, n_components=3000, C=10.0)
    model_pure.fit(X_train, y_train)
    pure_time = cpu_time() - start

    pure_accs = []
    for _ in range(reps):
        s = cpu_time()
        pred = model_pure.predict(X_test)
        pure_accs.append(accuracy_score(y_test, pred))
    pure_acc = np.mean(pure_accs)
    pure_cpu = pure_time / reps
    print(f"PURE: {pure_cpu:.3f}s | Acc: {pure_acc:.4f}")

    # --- XGBOOST ---
    print("\nTraining XGBOOST...")
    start = cpu_time()
    model_xgb = XGBoostModel()
    model_xgb.fit(X_train, y_train)
    xgb_time = cpu_time() - start

    xgb_accs = []
    for _ in range(reps):
        s = cpu_time()
        pred = model_xgb.predict(X_test)
        xgb_accs.append(accuracy_score(y_test, pred))
    xgb_acc = np.mean(xgb_accs)
    xgb_cpu = xgb_time / reps
    print(f"XGB: {xgb_cpu:.3f}s | Acc: {xgb_acc:.4f}")

    # --- Verdict ---
    speedup = xgb_cpu / pure_cpu if pure_cpu > 0 else float('inf')
    acc_diff = pure_acc - xgb_acc
    winner = "PURE ONESTEP" if pure_cpu < xgb_cpu and pure_acc >= xgb_acc else "XGBOOST"

    print(f"\nSPEEDUP: PURE ONESTEP {speedup:.2f}x faster")
    print(f"ACC DIFF: PURE ONESTEP {'+' if acc_diff >= 0 else ''}{acc_diff:.4f}")
    print(f"WINNER: {winner} WINS!")

    content = f"""PURE ONESTEP vs XGBOOST - 100K SAMPLES
PURE ONESTEP: {pure_cpu:.3f}s, Acc: {pure_acc:.4f}
XGBOOST: {xgb_cpu:.3f}s, Acc: {xgb_acc:.4f}
Speedup: {speedup:.2f}x | Acc Diff: {acc_diff:.4f}
Winner: {winner}"""
    save(content)

if __name__ == "__main__":
    main()
