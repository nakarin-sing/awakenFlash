#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py — Stable Streaming Benchmark
=====================================================
✅ แก้บั๊ก partial_fit class mismatch
✅ ป้องกัน overflow / NaN ใน RLS++
✅ รองรับ NumPy >= 1.25
✅ ใช้ dataset streaming จริงจาก UCI
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import psutil, time, urllib.request, gzip, io
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


# ========================
# 🔧 RLS++ Online Learner
# ========================

class RLSPlus:
    def __init__(self, n_features, lam=0.99, delta=1.0):
        self.lam = lam
        self.w = np.zeros((n_features, 1))
        self.P = np.eye(n_features) / delta

    def partial_fit(self, X, y):
        for i in range(X.shape[0]):
            xi = X[i, :].reshape(-1, 1)
            yi = y[i]

            # --- Stable update ---
            try:
                pred = (xi.T @ self.w).item()
            except Exception:
                pred = float(np.dot(xi.ravel(), self.w.ravel()))

            err = yi - pred
            denom = self.lam + xi.T @ self.P @ xi
            if denom <= 0 or np.isnan(denom):
                denom = self.lam + 1e-8

            k = (self.P @ xi) / denom
            self.w += k * err

            # Stable update for P
            self.P = (self.P - k @ xi.T @ self.P) / self.lam
            np.clip(self.P, -1e6, 1e6, out=self.P)
            if np.any(np.isnan(self.P)):
                self.P = np.eye(self.P.shape[0]) / 1.0  # reset safeguard

    def predict(self, X):
        y_pred = X @ self.w
        return (y_pred.ravel() > 0).astype(int)


# ==============================
# 📦 Dataset Stream Preparation
# ==============================

def load_covtype_stream(chunk_size=20000, max_chunks=5):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"🔗 Loading dataset streaming from: [{url}]")

    with urllib.request.urlopen(url) as f:
        data = pd.read_csv(io.BytesIO(gzip.decompress(f.read())), header=None)

    X = data.iloc[:, :-1].values.astype(np.float32)
    y = (data.iloc[:, -1].values == 2).astype(int)  # binary simplify

    X, y = shuffle(X, y, random_state=42)

    for i in range(0, min(len(X), chunk_size * max_chunks), chunk_size):
        yield X[i:i + chunk_size], y[i:i + chunk_size]


# =======================
# 🚀 Benchmark Function
# =======================

def benchmark():
    print("🚀 Starting benchmark...\n")
    stream = load_covtype_stream()

    # Prepare models
    sgd = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", tol=None)
    rls = None
    xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.3, tree_method="hist", verbosity=0)

    scaler = StandardScaler()
    all_classes = np.array([0, 1])  # fix class list for partial_fit

    results = []

    for i, (X, y) in enumerate(stream, start=1):
        print(f"===== Processing Chunk {i:02d} =====")

        if i == 1:
            scaler.fit(X)
            X = scaler.transform(X)
            rls = RLSPlus(n_features=X.shape[1])
        else:
            X = scaler.transform(X)

        # --- SGD ---
        t0 = time.time()
        sgd.partial_fit(X, y, classes=all_classes)
        pred_sgd = sgd.predict(X)
        acc_sgd = accuracy_score(y, pred_sgd)
        t1 = time.time() - t0

        # --- RLS++ ---
        t0 = time.time()
        rls.partial_fit(X, y)
        pred_rls = rls.predict(X)
        acc_rls = accuracy_score(y, pred_rls)
        t2 = time.time() - t0

        # --- XGBoost ---
        t0 = time.time()
        xgb_model.fit(X, y)
        pred_xgb = xgb_model.predict(X)
        acc_xgb = accuracy_score(y, pred_xgb)
        t3 = time.time() - t0

        print(f"SGD: acc={acc_sgd:.3f}, time={t1:.3f}s")
        print(f"RLS++: acc={acc_rls:.3f}, time={t2:.3f}s")
        print(f"XGB: acc={acc_xgb:.3f}, time={t3:.3f}s\n")

        results.append([i, acc_sgd, acc_rls, acc_xgb, t1, t2, t3])

    df = pd.DataFrame(results, columns=["Chunk", "SGD_acc", "RLS_acc", "XGB_acc", "SGD_time", "RLS_time", "XGB_time"])
    df.to_csv("benchmark_results/awakenFlash_results.csv", index=False)
    print("✅ Benchmark complete → saved to benchmark_results/awakenFlash_results.csv")


if __name__ == "__main__":
    import os
    os.makedirs("benchmark_results", exist_ok=True)
    benchmark()
