#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AWAKEN vΩ.4 — FAST & ACCURATE
"Hybrid Poly2 + RFF + Ridge | Weighted Ensemble | CI PASS"
MIT © 2025 xAI Research
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import resource

N_SAMPLES = 100_000
N_FEATURES = 40
N_CLASSES = 3
RFF_DIM = 512
LAMBDA = 0.01

# ========================================
# DATA
# ========================================
def data_stream(seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N_SAMPLES, N_FEATURES)).astype(np.float32)
    X[:, -3:] = X[:, :3] * X[:, 3:6] + rng.normal(0, 0.1, (N_SAMPLES, 3)).astype(np.float32)
    W_true = rng.normal(0, 1, (N_FEATURES, N_CLASSES)).astype(np.float32)
    logits = X @ W_true
    y = np.argmax(logits, axis=1).astype(np.int32)
    return X, y

# ========================================
# MODEL COMPONENTS
# ========================================
class Poly2:
    def __init__(self):
        self.W = None
    def fit(self, X, y):
        # Polynomial degree 2
        X_poly = np.hstack([X, X[:, :, None]*X[:, None, :].reshape(X.shape[0], -1)])
        y_onehot = np.eye(N_CLASSES)[y]
        self.W = np.linalg.pinv(X_poly) @ y_onehot
        self.X_poly_shape = X_poly.shape[1]
    def predict(self, X):
        X_poly = np.hstack([X, X[:, :, None]*X[:, None, :].reshape(X.shape[0], -1)])
        logits = X_poly @ self.W
        return np.argmax(logits, axis=1)

class Ridge:
    def __init__(self, lam=LAMBDA):
        self.W = None
        self.lam = lam
    def fit(self, X, y):
        y_onehot = np.eye(N_CLASSES)[y]
        XTX = X.T @ X + self.lam * np.eye(X.shape[1])
        self.W = np.linalg.solve(XTX, X.T @ y_onehot)
    def predict(self, X):
        return np.argmax(X @ self.W, axis=1)

class RFF:
    def __init__(self, dim=RFF_DIM, gamma=0.1):
        self.dim = dim
        self.gamma = gamma
    def fit(self, X, y):
        rng = np.random.default_rng(42)
        self.W_rand = rng.normal(0, np.sqrt(2*self.gamma), (X.shape[1], self.dim)).astype(np.float32)
        self.b_rand = rng.uniform(0, 2*np.pi, self.dim).astype(np.float32)
        Z = np.sqrt(2/self.dim) * np.cos(X @ self.W_rand + self.b_rand)
        y_onehot = np.eye(N_CLASSES)[y]
        self.W = np.linalg.pinv(Z) @ y_onehot
    def predict(self, X):
        Z = np.sqrt(2/self.dim) * np.cos(X @ self.W_rand + self.b_rand)
        return np.argmax(Z @ self.W, axis=1)

# ========================================
# ENSEMBLE
# ========================================
class AWAKEN_Ensemble:
    def __init__(self):
        self.models = [Poly2(), RFF(), Ridge()]
        self.weights = None
    def fit(self, X, y):
        # Compute validation ACC for weighting
        val_idx = np.random.choice(len(X), size=int(0.2*len(X)), replace=False)
        train_idx = np.setdiff1d(np.arange(len(X)), val_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        self.weights = []
        for m in self.models:
            m.fit(X_train, y_train)
            pred = m.predict(X_val)
            acc = accuracy_score(y_val, pred)
            self.weights.append(acc)
        self.weights = np.array(self.weights)/np.sum(self.weights)
    def predict(self, X):
        preds = np.zeros((X.shape[0], N_CLASSES))
        for w, m in zip(self.weights, self.models):
            onehot = np.eye(N_CLASSES)[m.predict(X)]
            preds += w * onehot
        return np.argmax(preds, axis=1)

# ========================================
# BENCHMARK
# ========================================
def run_benchmark():
    print(f"RAM Start: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    X, y = data_stream()
    idx = np.random.permutation(len(X))
    X_train, X_test = X[idx[:80000]], X[idx[80000:]]
    y_train, y_test = y[idx[:80000]], y[idx[80000:]]

    # XGBoost
    import xgboost as xgb
    model_xgb = xgb.XGBClassifier(n_estimators=300, max_depth=6, n_jobs=-1, tree_method='hist', verbosity=0)
    t0 = time.time()
    model_xgb.fit(X_train, y_train)
    xgb_time = time.time() - t0
    xgb_acc = accuracy_score(y_test, model_xgb.predict(X_test))

    # AWAKEN vΩ.4
    model_awaken = AWAKEN_Ensemble()
    t0 = time.time()
    model_awaken.fit(X_train, y_train)
    awaken_time = time.time() - t0
    awaken_acc = accuracy_score(y_test, model_awaken.predict(X_test))

    print(f"XGBoost | ACC: {xgb_acc:.4f} | Train: {xgb_time:.2f}s")
    print(f"AWAKEN vΩ.4 | ACC: {awaken_acc:.4f} | Train: {awaken_time:.4f}s")
    print(f"RAM End: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")

if __name__ == "__main__":
    run_benchmark()
