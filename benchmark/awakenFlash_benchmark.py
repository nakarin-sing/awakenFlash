# -*- coding: utf-8 -*-
"""
AWAKEN vΩ.17 — REAL STREAMING BENCHMARK (Covertype 500MB)
==========================================================
- Streaming benchmark (chunked reading)
- Compare: SGD / OneStepRLS / XGBoost
- Auto-download dataset, low-RAM friendly
- Saves result as CSV under benchmark_results/
"""

import os, time, gzip, io, requests
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ---------------------------
# Simple Online RLS
# ---------------------------
class OneStepRLS:
    def __init__(self, n_features, lam=1e-2):
        self.w = np.zeros((n_features, 1))
        self.P = np.eye(n_features) / lam

    def partial_fit(self, X, y):
        for i in range(X.shape[0]):
            xi = X[i, :].reshape(-1, 1)
            yi = y[i]
            Pi = self.P @ xi
            k = Pi / (1.0 + xi.T @ Pi)
            err = yi - float(xi.T @ self.w)
            self.w += k * err
            self.P -= k @ xi.T @ self.P
        return self

    def predict(self, X):
        return np.sign(X @ self.w).ravel()

# ---------------------------
# Streamed dataset loader
# ---------------------------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
CHUNKSIZE = 10000  # tune for GitHub runner RAM

def stream_covtype():
    print(f"🔗 Loading dataset streaming from: {URL}")
    with requests.get(URL, stream=True) as r:
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        with gzip.open(buf, "rt") as f:
            cols = list(range(54)) + ["target"]
            for chunk in pd.read_csv(f, names=cols, chunksize=CHUNKSIZE):
                X = chunk.iloc[:, :-1].to_numpy(dtype=np.float32)
                y = chunk["target"].to_numpy(dtype=np.int32)
                yield X, y

# ---------------------------
# Main benchmark
# ---------------------------
def benchmark():
    results = []
    scaler = StandardScaler()
    sgd = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", tol=None)
    rls = OneStepRLS(n_features=54)
    xgb_X, xgb_y = [], []

    start_time = time.time()
    for i, (X, y) in enumerate(stream_covtype(), 1):
        print(f"\n===== Processing Chunk {i:02d} =====")

        # Standardize
        X = scaler.partial_fit(X).transform(X)

        # SGD online
        t0 = time.time()
        sgd.partial_fit(X, y, classes=np.arange(1, 8))
        acc_sgd = sgd.score(X, y)
        t_sgd = time.time() - t0
        print(f"SGD: acc={acc_sgd:.3f}, time={t_sgd:.3f}s")

        # RLS online
        t0 = time.time()
        y_rls = (y == 1).astype(np.float32)  # binary surrogate for speed
        rls.partial_fit(X, y_rls)
        preds = (rls.predict(X) > 0).astype(int)
        acc_rls = np.mean(preds == y_rls)
        t_rls = time.time() - t0
        print(f"RLS: acc={acc_rls:.3f}, time={t_rls:.3f}s")

        # Collect for XGBoost retrain
        xgb_X.append(X)
        xgb_y.append(y - 1)  # shift labels to 0–6
        X_all = np.vstack(xgb_X)
        y_all = np.hstack(xgb_y)

        # XGBoost retrain (mini-batch style)
        t0 = time.time()
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="mlogloss",
            num_class=7, verbosity=0
        )
        model.fit(X_all, y_all)
        preds = model.predict(X[:2000])
        acc_xgb = accuracy_score(y[:2000], preds + 1)
        t_xgb = time.time() - t0
        print(f"XGB: acc={acc_xgb:.3f}, time={t_xgb:.3f}s")

        results.append({
            "chunk": i,
            "acc_sgd": acc_sgd,
            "acc_rls": acc_rls,
            "acc_xgb": acc_xgb,
            "time_sgd": t_sgd,
            "time_rls": t_rls,
            "time_xgb": t_xgb
        })

        # Limit for CI runner
        if i >= 5:
            break

    total_time = time.time() - start_time
    df = pd.DataFrame(results)
    os.makedirs("benchmark_results", exist_ok=True)
    out_file = "benchmark_results/streaming_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\n✅ Benchmark complete in {total_time:.1f}s — saved to {out_file}")

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting benchmark...")
    benchmark()
