#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AbsoluteNon Online Transformer + XGBoost Streaming Hybrid
Adaptive folding per batch for online incremental learning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ================= AbsoluteNon Online Transformer ===================
class AbsoluteNonOnline:
    def __init__(self, n=5, alpha=0.7):
        self.n = n
        self.alpha = alpha
        self.centers = None
        self.fitted = False

    def update_centers(self, X_batch):
        median_batch = np.median(X_batch, axis=0)
        if self.centers is None:
            self.centers = median_batch
        else:
            # moving average update
            self.centers = 0.9 * self.centers + 0.1 * median_batch
        self.fitted = True

    def transform(self, X_batch):
        if not self.fitted:
            raise ValueError("Centers not initialized. Call update_centers first.")
        X_folded = X_batch.copy()
        for _ in range(self.n):
            X_folded = self.alpha * (0.5 - np.abs(X_folded - self.centers)) + (1 - self.alpha) * X_folded
        return X_folded

    def fit_transform(self, X_batch):
        self.update_centers(X_batch)
        return self.transform(X_batch)

# ================= Dataset Loading Example ===================
def load_covtype_sample(n_samples=50000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    df = pd.read_csv(url, header=None, nrows=n_samples)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values - 1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# ================= Streaming Hybrid Benchmark ===================
def streaming_hybrid():
    X, y = load_covtype_sample()
    chunk_size = 5000
    window_size = 5  # XGBoost rolling window
    xgb_X_chunks, xgb_y_chunks = [], []

    transformer = AbsoluteNonOnline(n=5, alpha=0.7)
    acc_list = []

    for i in range(0, len(X), chunk_size):
        X_batch = X[i:i+chunk_size]
        y_batch = y[i:i+chunk_size]

        # --- Transform Batch ---
        X_folded = transformer.fit_transform(X_batch)

        # --- Rolling window for XGBoost ---
        xgb_X_chunks.append(X_folded)
        xgb_y_chunks.append(y_batch)
        if len(xgb_X_chunks) > window_size:
            xgb_X_chunks = xgb_X_chunks[-window_size:]
            xgb_y_chunks = xgb_y_chunks[-window_size:]

        X_train = np.vstack(xgb_X_chunks[:-1]) if len(xgb_X_chunks) > 1 else X_folded
        y_train = np.concatenate(xgb_y_chunks[:-1]) if len(xgb_y_chunks) > 1 else y_batch
        X_test = X_folded
        y_test = y_batch

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(
            {"objective": "multi:softmax", "num_class": 7, "max_depth": 5,
             "eta": 0.1, "subsample": 0.8, "verbosity": 0},
            dtrain, num_boost_round=20
        )
        y_pred = model.predict(dtest)
        acc = accuracy_score(y_test, y_pred)
        acc_list.append(acc)
        print(f"Chunk {i//chunk_size+1:02d}: Accuracy={acc:.4f}")

    print(f"\n✅ Streaming Hybrid Average Accuracy: {np.mean(acc_list):.4f}")

if __name__ == "__main__":
    streaming_hybrid()
