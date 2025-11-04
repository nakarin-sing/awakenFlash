#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py — Stable Real-World Benchmark
======================================================
Benchmark: SGD vs SRLS++ vs XGBoost
Dataset : UCI Covertype (streaming by chunk)
"""

import numpy as np
import pandas as pd
import gzip, urllib.request, time
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 1. Stabilized Recursive Least Squares
# ======================================================

class SRLSpp:
    def __init__(self, n_features, lam=0.99, delta=1.0, clip=5.0, lr_decay=1e-4):
        self.lam = lam
        self.delta = delta
        self.clip = clip
        self.lr_decay = lr_decay
        self.P = np.eye(n_features) / delta
        self.w = np.zeros((n_features, 1))
        self.step = 1

    def update(self, x, y):
        x = x.reshape(-1, 1)
        Px = self.P @ x
        gain = Px / (self.lam + x.T @ Px)
        pred = (self.w.T @ x).item()
        err = np.clip(y - pred, -self.clip, self.clip)

        eta = 1.0 / (1.0 + self.lr_decay * self.step)
        self.w += eta * (gain * err)
        self.P = (self.P - gain @ x.T @ self.P) / self.lam
        self.step += 1

    def partial_fit(self, X, y):
        for xi, yi in zip(X, y):
            self.update(xi, yi)

    def predict(self, X):
        return (X @ self.w).ravel()


# ======================================================
# 2. Streaming Dataset Loader (Covertype)
# ======================================================

def stream_covtype(url, chunksize=5000):
    print(f"🔗 Loading dataset streaming from: [[{url}]]")
    with urllib.request.urlopen(url) as f:
        with gzip.GzipFile(fileobj=f) as gz:
            data = pd.read_csv(gz, header=None)
            X = data.iloc[:, :-1].values
            y = (data.iloc[:, -1] == 2).astype(int).values  # binary
            for i in range(0, len(X), chunksize):
                yield X[i:i+chunksize], y[i:i+chunksize]


# ======================================================
# 3. Main Benchmark
# ======================================================

def run_benchmark():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    results = []

    chunk_id = 0
    for X_chunk, y_chunk in stream_covtype(url):
        chunk_id += 1
        scaler = StandardScaler()
        X_chunk = scaler.fit_transform(X_chunk)
        n_features = X_chunk.shape[1]

        # --- SGD baseline ---
        sgd = SGDClassifier(max_iter=5, tol=1e-3)
        t0 = time.time()
        sgd.fit(X_chunk, y_chunk)
        sgd_pred = sgd.predict(X_chunk)
        sgd_acc = accuracy_score(y_chunk, sgd_pred)
        t_sgd = time.time() - t0

        # --- SRLS++ ---
        srls = SRLSpp(n_features=n_features)
        t0 = time.time()
        srls.partial_fit(X_chunk, y_chunk)
        srls_pred = (srls.predict(X_chunk) > 0.5).astype(int)
        srls_acc = accuracy_score(y_chunk, srls_pred)
        t_srls = time.time() - t0

        # --- XGBoost ---
        xgb = XGBClassifier(
            n_estimators=25,
            max_depth=5,
            learning_rate=0.3,
            subsample=0.7,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        t0 = time.time()
        xgb.fit(X_chunk, y_chunk)
        xgb_pred = xgb.predict(X_chunk)
        xgb_acc = accuracy_score(y_chunk, xgb_pred)
        t_xgb = time.time() - t0

        print(f"\n===== Processing Chunk {chunk_id:02d} =====")
        print(f"SGD:   acc={sgd_acc:.3f}, time={t_sgd:.3f}s")
        print(f"SRLS++: acc={srls_acc:.3f}, time={t_srls:.3f}s")
        print(f"XGB:   acc={xgb_acc:.3f}, time={t_xgb:.3f}s")

        results.append({
            "chunk": chunk_id,
            "SGD_acc": sgd_acc,
            "SGD_time": t_sgd,
            "SRLS_acc": srls_acc,
            "SRLS_time": t_srls,
            "XGB_acc": xgb_acc,
            "XGB_time": t_xgb,
        })

        if chunk_id >= 5:
            break

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results/awakenFlash_results.csv", index=False)
    print("\n✅ Benchmark complete → saved to benchmark_results/awakenFlash_results.csv")

    # summary
    avg = df.mean(numeric_only=True)
    print("\n📊 Average Performance Summary:")
    print(f"  SGD   — acc={avg['SGD_acc']:.3f}, time={avg['SGD_time']:.3f}s")
    print(f"  SRLS++ — acc={avg['SRLS_acc']:.3f}, time={avg['SRLS_time']:.3f}s")
    print(f"  XGB   — acc={avg['XGB_acc']:.3f}, time={avg['XGB_time']:.3f}s")

    best = max(avg['SGD_acc'], avg['SRLS_acc'], avg['XGB_acc'])
    if best == avg['SRLS_acc']:
        winner = "SRLS++"
    elif best == avg['XGB_acc']:
        winner = "XGB"
    else:
        winner = "SGD"

    print(f"\n🏁 Winner: {winner} (mean acc={best:.3f})")


if __name__ == "__main__":
    run_benchmark()
