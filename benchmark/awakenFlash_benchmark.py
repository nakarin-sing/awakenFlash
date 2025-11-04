#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py
Real-world streaming benchmark for AwakenFlash
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb


def benchmark():
    print("🚀 Starting benchmark...")

    # === 1. Load dataset ===
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"🔗 Loading dataset streaming from: {url}")

    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1  # Convert to 0–6

    # All classes for partial_fit
    all_classes = np.unique(y_all)

    # Split into chunks to simulate streaming
    chunk_size = 10000
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) for i in range(0, len(X_all), chunk_size)]

    # === 2. Init models ===
    sgd = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
    rls = RidgeClassifier()
    xgb_model = None  # will init later

    first_sgd = True

    os.makedirs("benchmark_results", exist_ok=True)

    # === 3. Stream loop ===
    for chunk_id, (X, y) in enumerate(chunks[:5], start=1):  # limit 5 chunks for speed
        print(f"\n===== Processing Chunk {chunk_id:02d} =====")

        # SGD (online)
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X, y, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X, y)
        sgd_acc = sgd.score(X, y)
        sgd_time = time.time() - start
        print(f"SGD: acc={sgd_acc:.3f}, time={sgd_time:.3f}s")

        # Ridge (batch)
        start = time.time()
        rls.fit(X, y)
        rls_acc = rls.score(X, y)
        rls_time = time.time() - start
        print(f"RLS: acc={rls_acc:.3f}, time={rls_time:.3f}s")

        # XGBoost (mini batch training)
        start = time.time()
        dtrain = xgb.DMatrix(X, label=y)
        if xgb_model is None:
            xgb_model = xgb.train(
                {"objective": "multi:softmax", "num_class": 7, "verbosity": 0},
                dtrain,
                num_boost_round=10,
            )
        else:
            xgb_model = xgb.train(
                {"objective": "multi:softmax", "num_class": 7, "verbosity": 0},
                dtrain,
                num_boost_round=5,
                xgb_model=xgb_model,
            )
        preds = xgb_model.predict(dtrain)
        xgb_acc = accuracy_score(y, preds)
        xgb_time = time.time() - start
        print(f"XGB: acc={xgb_acc:.3f}, time={xgb_time:.3f}s")

        # Save per-chunk results
        with open("benchmark_results/chunk_log.txt", "a") as f:
            f.write(
                f"Chunk {chunk_id:02d}: SGD={sgd_acc:.3f}, RLS={rls_acc:.3f}, XGB={xgb_acc:.3f}\n"
            )

    print("\n✅ Benchmark finished successfully.")
    print("Results saved to benchmark_results/chunk_log.txt")


if __name__ == "__main__":
    benchmark()
