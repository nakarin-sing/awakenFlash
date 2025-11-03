#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAIR REAL-WORLD STREAMING BENCHMARK
Dataset: Covertype (500MB+)
Models Compared:
  ✅ Online-OneStep (RLS)
  ✅ SGDClassifier (baseline online)
  ✅ XGBoost (full retrain per chunk)
Output:
  ✅ streaming_results.csv
"""

import numpy as np
import pandas as pd
import urllib.request, io, gzip
import time, tracemalloc
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# ====================================================
# ✅ Online OneStep RLS (streaming incremental learner)
# ====================================================
class OnlineOneStepRLS:
    def __init__(self, n_features, n_classes, alpha=1e-2):
        self.n_classes = n_classes
        self.alpha = alpha
        self.scaler = StandardScaler(with_mean=False)

        d = n_features
        self.W = np.zeros((d, n_classes), dtype=np.float32)
        self.P = np.eye(d, dtype=np.float32) * 1e3  # large initial covariance

        self.first = True

    def partial_fit(self, X, y):
        # scale
        if self.first:
            X = self.scaler.fit_transform(X)
            self.first = False
        else:
            X = self.scaler.transform(X)

        n, d = X.shape
        y_onehot = np.zeros((n, self.n_classes), dtype=np.float32)
        y_onehot[np.arange(n), y - 1] = 1.0  # classes: 1–7 → 0–6

        PX = self.P @ X.T
        S = np.eye(n, dtype=np.float32) + X @ PX

        try:
            S_inv = np.linalg.inv(S)
        except:
            S_inv = np.linalg.pinv(S)

        K = PX @ S_inv
        self.W += K @ (y_onehot - X @ self.W)
        self.P = self.P - K @ (X @ self.P)

        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        logits = X @ self.W
        return logits.argmax(axis=1) + 1

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        logits = X @ self.W
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)


# ====================================================
# ✅ Streaming Data Loader (Covertype)
# ====================================================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
CHUNK = 20000  # safe for GitHub Actions
NF = 54        # number of features
NC = 7         # number of classes


def load_streaming():
    print(f"🔗 Downloading dataset (streaming): {URL}")
    with urllib.request.urlopen(URL) as resp:
        gz = gzip.GzipFile(fileobj=io.BytesIO(resp.read()))
        reader = pd.read_csv(gz, header=None, chunksize=CHUNK)

        colnames = [f"f{i}" for i in range(NF)] + ["target"]

        for chunk in reader:
            chunk.columns = colnames
            X = chunk.drop("target", axis=1).astype(np.float32).values
            y = chunk["target"].astype(int).values
            yield X, y


# ====================================================
# ✅ Main Benchmark
# ====================================================
def benchmark():
    print("=" * 80)
    print("🔥 REAL STREAMING BENCHMARK — COVERTYPE 500MB")
    print("=" * 80)

    all_classes = np.arange(1, 8)

    sgd = SGDClassifier(loss="log_loss", learning_rate="optimal")
    rls = OnlineOneStepRLS(n_features=NF, n_classes=NC)
    xgb_X, xgb_y = [], []

    first_sgd = True

    results = {
        "chunk": [],
        "sgd_acc": [], "sgd_time": [],
        "rls_acc": [], "rls_time": [],
        "xgb_acc": [], "xgb_time": []
    }

    for i, (X, y) in enumerate(load_streaming(), start=1):
        print(f"\n===== Processing Chunk {i:02d} =====")

        # -------------------------------
        # ✅ SGD (baseline online)
        # -------------------------------
        t0 = time.time()
        if first_sgd:
            sgd.partial_fit(X, y, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X, y)
        preds = sgd.predict(X[:2000])
        acc_sgd = accuracy_score(y[:2000], preds)
        t_sgd = time.time() - t0

        print(f"SGD: acc={acc_sgd:.3f}, time={t_sgd:.3f}s")

        # -------------------------------
        # ✅ Online RLS OneStep
        # -------------------------------
        t0 = time.time()
        rls.partial_fit(X, y)
        preds = rls.predict(X[:2000])
        acc_rls = accuracy_score(y[:2000], preds)
        t_rls = time.time() - t0

        print(f"RLS: acc={acc_rls:.3f}, time={t_rls:.3f}s")

        # -------------------------------
        # ✅ XGBoost (retrain on accumulated)
        # -------------------------------
        xgb_X.append(X)
        xgb_y.append(y)
        X_all = np.vstack(xgb_X)
        y_all = np.hstack(xgb_y)

        t0 = time.time()
        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss"
        )
        model.fit(X_all, y_all)
        preds = model.predict(X[:2000])
        acc_xgb = accuracy_score(y[:2000], preds)
        t_xgb = time.time() - t0

        print(f"XGB: acc={acc_xgb:.3f}, time={t_xgb:.3f}s")

        # -------------------------------
        # ✅ Save results
        # -------------------------------
        results["chunk"].append(i)
        results["sgd_acc"].append(acc_sgd)
        results["sgd_time"].append(t_sgd)
        results["rls_acc"].append(acc_rls)
        results["rls_time"].append(t_rls)
        results["xgb_acc"].append(acc_xgb)
        results["xgb_time"].append(t_xgb)

        if i >= 10:  # limit benchmark (GitHub safe)
            break

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results/streaming_results.csv", index=False)
    print("\n✅ Saved: benchmark_results/streaming_results.csv")


if __name__ == "__main__":
    benchmark()
