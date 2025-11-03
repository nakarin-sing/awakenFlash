#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.16 — STREAMING MICROENSEMBLE LOW-RAM
"Mini-batch RLS online update, minimal memory footprint."
MIT © 2025 xAI Research
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import resource

# ========================================
# STREAMING ONE-STEP MICROENSEMBLE
# ========================================
class StreamingOneStep:
    """
    Online RLS with minimal quadratic expansion per batch
    """
    def __init__(self, n_features, C=1e-3):
        self.C = C
        self.n_features = n_features
        self.n_params = 1 + n_features + n_features  # bias + linear + quadratic
        self.W = np.zeros((self.n_params,))          # initial weights
        self.P = np.eye(self.n_params, dtype=np.float32) / self.C  # RLS covariance

    def _expand_features(self, X):
        # X: batch_size x n_features
        X_b = np.hstack([np.ones((X.shape[0],1), dtype=np.float32), X])
        X_quad = X**2
        return np.hstack([X_b, X_quad])

    def update_batch(self, X_batch, y_batch):
        # Convert labels to one-hot
        K = y_batch.max() + 1
        Y_onehot = np.eye(K, dtype=np.float32)[y_batch]

        X_feat = self._expand_features(X_batch)  # batch_size x n_params
        for xi, yi in zip(X_feat, Y_onehot):
            xi = xi.reshape(-1,1)
            P_x = self.P @ xi
            gain = P_x / (1.0 + xi.T @ P_x)
            e = yi - self.W @ xi.ravel()
            self.W += (gain.ravel() * e)
            self.P = self.P - gain @ xi.T @ self.P

    def predict(self, X):
        X_feat = self._expand_features(X)
        logits = X_feat @ self.W
        return logits.round().astype(int)  # simple rounding for multiclass

# ========================================
# BENCHMARK
# ========================================
def benchmark_streaming():
    print(f"RAM Start: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")

    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]

    xgb_config = dict(n_estimators=50, max_depth=4, n_jobs=1, verbosity=0, tree_method='hist')

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        X = (X - X.mean(axis=0)) / X.std(axis=0)  # simple scaling

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- XGBoost ---
        t0 = time.time()
        model = xgb.XGBClassifier(**xgb_config)
        model.fit(X_train, y_train)
        t_xgb = time.time() - t0
        pred_xgb = model.predict(X_test)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        f1_xgb = f1_score(y_test, pred_xgb, average='weighted')

        # --- Streaming MicroEnsemble ---
        n_features = X_train.shape[1]
        s_model = StreamingOneStep(n_features)
        batch_size = 16
        t0 = time.time()
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            s_model.update_batch(X_batch, y_batch)
        t_stream = time.time() - t0
        pred_stream = s_model.predict(X_test)
        acc_stream = accuracy_score(y_test, pred_stream)
        f1_stream = f1_score(y_test, pred_stream, average='weighted')

        print(f"\n===== {name} =====")
        print(f"{'Model':<15} {'ACC':<8} {'F1':<8} {'Time':<8} {'RAM'}")
        ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"{'XGBoost':<15} {acc_xgb:.4f}   {f1_xgb:.4f}   {t_xgb:.4f}s  {ram:.1f} MB")
        print(f"{'StreamingOS':<15} {acc_stream:.4f}   {f1_stream:.4f}   {t_stream:.4f}s  {ram:.1f} MB")

    print(f"\nRAM End: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    print("="*60)
    print("AWAKEN vΩ.16 — STREAMING LOW-RAM MICROENSEMBLE BENCHMARK")
    print("="*60)

if __name__ == "__main__":
    benchmark_streaming()
