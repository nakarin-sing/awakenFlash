#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py — Adaptive SRLS Benchmark
===================================================
SGD vs A-SRLS vs XGBoost (streaming chunks)
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
# Adaptive Stabilized Recursive Least Squares
# ======================================================

class ASRLS:
    def __init__(self, n_features, lam_base=0.98, delta=1.0, lr_decay=1e-4, alpha=0.2):
        self.lam_base = lam_base
        self.delta = delta
        self.lr_decay = lr_decay
        self.alpha = alpha
        self.P = np.eye(n_features) / delta
        self.w = np.zeros((n_features, 1))
        self.step = 1
        self.err_hist = []

    def _adaptive_lambda(self):
        if len(self.err_hist) < 10:
            return self.lam_base
        var_e = np.var(self.err_hist[-20:])
        lam_t = self.lam_base + self.alpha * (var_e / (1 + var_e))
        return float(np.clip(lam_t, 0.90, 0.999))

    def update(self, x, y):
        x = x.reshape(-1, 1)
        Px = self.P @ x
        lam_t = self._adaptive_lambda()
        gain = Px / (lam_t + x.T @ Px)

        pred = (self.w.T @ x).item()
        err = y - pred
        self.err_hist.append(err)
        clip = np.median(np.abs(self.err_hist[-50:])) * 3.0 + 1e-8
        err = np.clip(err, -clip, clip)

        eta = 1.0 / (1.0 + self.lr_decay * self.step)
        self.w += eta * (gain * err)
        self.P = (self.P - gain @ x.T @ self.P) / lam_t
        self.step += 1

    def partial_fit(self, X, y):
        for xi, yi in zip(X, y):
            self.update(xi, yi)

    def predict(self, X):
        return (X @ self.w).ravel()


# ======================================================
# Streaming dataset loader
# ======================================================

def stream_covtype(url, chunksize=5000):
    print(f"🔗 Loading dataset streaming from: [[{url}]]")
    with urllib.request.urlopen(url) as f:
        with gzip.GzipFile(fileobj=f) as gz:
            data = pd.read_csv(gz, header=None)
            X = data.iloc[:, :-1].values
            y = (data.iloc[:, -1] == 2).astype(int).values
            for i in range(0, len(X), chunksize):
                yield X[i:i+chunksize], y[i:i+chunksize]


# ======================================================
# Main benchmark
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

        # SGD baseline
        sgd = SGDClassifier(max_iter=5, tol=1e-3)
        t0 = time.time()
        sgd.fit(X_chunk, y_chunk)
        sgd_pred = sgd.predict(X_chunk)
        sgd_acc = accuracy_score(y_chunk, sgd_pred)
        t_sgd = time.time() - t0

        # A-SRLS
        srls = ASRLS(n_features=n_features)
        t0 = time.time()
        srls.partial_fit(X_chunk, y_chunk)
        srls_pred = (srls.predict(X_chunk) > 0.5).astype(int)
        srls_acc = accuracy_score(y_chunk, srls_pred)
        t_srls = time.time() - t0

        # XGBoost
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
        print(f"A-SRLS: acc={srls_acc:.3f}, time={t_srls:.3f}s")
        print(f"XGB:   acc={xgb_acc:.3f}, time={t_xgb:.3f}s")

        results.append({
            "chunk": chunk_id,
            "SGD_acc": sgd_acc,
            "SGD_time": t_sgd,
            "ASRLS_acc": srls_acc,
            "ASRLS_time": t_srls,
            "XGB_acc": xgb_acc,
            "XGB_time": t_xgb,
        })

        if chunk_id >= 5:
            break

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results/awakenFlash_results.csv", index=False)
    print("\n✅ Benchmark complete → saved to benchmark_results/awakenFlash_results.csv")

    avg = df.mean(numeric_only=True)
    print("\n📊 Average Performance Summary:")
    print(f"  SGD   — acc={avg['SGD_acc']:.3f}, time={avg['SGD_time']:.3f}s")
    print(f"  A-SRLS — acc={avg['ASRLS_acc']:.3f}, time={avg['ASRLS_time']:.3f}s")
    print(f"  XGB   — acc={avg['XGB_acc']:.3f}, time={avg['XGB_time']:.3f}s")

    best = max(avg['SGD_acc'], avg['ASRLS_acc'], avg['XGB_acc'])
    if best == avg['ASRLS_acc']:
        winner = "A-SRLS"
    elif best == avg['XGB_acc']:
        winner = "XGB"
    else:
        winner = "SGD"
    print(f"\n🏁 Winner: {winner} (mean acc={best:.3f})")


if __name__ == "__main__":
    run_benchmark()
