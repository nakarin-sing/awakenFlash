#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.16 — STREAMING ULTIMATE CORE
"Online / Micro-Batch RLS with Minimal RAM and Multi-Class Support"
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
# STREAMING ONESTEP ULTIMATE CORE
# ========================================
class StreamingOneStep:
    def __init__(self, n_features, n_classes, C=1e-3):
        self.C = C
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_params = 1 + n_features + n_features  # bias + linear + quadratic
        self.W = np.zeros((self.n_params, n_classes), dtype=np.float32)
        self.P = np.eye(self.n_params, dtype=np.float32) / self.C

    def _expand_features(self, X):
        X_b = np.hstack([np.ones((X.shape[0],1), dtype=np.float32), X])
        X_quad = X**2
        return np.hstack([X_b, X_quad])

    def update_batch(self, X_batch, y_batch):
        X_feat = self._expand_features(X_batch)
        for xi, yi in zip(X_feat, y_batch):
            xi = xi.reshape(-1,1)
            P_x = self.P @ xi
            gain = P_x / (1.0 + xi.T @ P_x)
            e = np.zeros((self.n_classes,), dtype=np.float32)
            e[yi] = 1.0
            e = e - (xi.T @ self.W).ravel()
            self.W += gain @ e.reshape(1,-1)
            self.P = self.P - gain @ xi.T @ self.P

    def predict(self, X):
        X_feat = self._expand_features(X)
        logits = X_feat @ self.W
        return logits.argmax(axis=1)


# ========================================
# OPTIMIZED STREAMING BENCHMARK
# ========================================
def benchmark_streaming(batch_size=16):
    print(f"RAM Start: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB\n")

    xgb_config = dict(n_estimators=50, max_depth=4, n_jobs=1, verbosity=0, tree_method='hist')
    
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # --- Streaming OneStep ---
        s_model = StreamingOneStep(n_features, n_classes)
        t0 = time.time()
        # Micro-batch updates
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            s_model.update_batch(X_batch, y_batch)
        t_stream = time.time() - t0
        pred_stream = s_model.predict(X_test)
        acc_stream = accuracy_score(y_test, pred_stream)
        f1_stream = f1_score(y_test, pred_stream, average='weighted')

        # --- XGBoost ---
        t0 = time.time()
        xgb_model = xgb.XGBClassifier(**xgb_config)
        xgb_model.fit(X_train, y_train)
        t_xgb = time.time() - t0
        pred_xgb = xgb_model.predict(X_test)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        f1_xgb = f1_score(y_test, pred_xgb, average='weighted')

        # Print results
        print(f"===== {name} =====")
        print(f"{'Model':<10} {'ACC':<8} {'F1':<8} {'Time(s)':<8} {'RAM(MB)':<8}")
        ram_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"{'Streaming':<10} {acc_stream:.4f} {f1_stream:.4f} {t_stream:.4f} {ram_used:.1f}")
        print(f"{'XGBoost':<10} {acc_xgb:.4f} {f1_xgb:.4f} {t_xgb:.4f} {ram_used:.1f}\n")

    print(f"RAM End: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    print("="*60)
    print("AWAKEN vΩ.16 — STREAMING ONESTEP BENCHMARK COMPLETE")
    print("Low-RAM, Multi-Class, Online Micro-Batch RLS")
    print("="*60)


if __name__ == "__main__":
    benchmark_streaming()
