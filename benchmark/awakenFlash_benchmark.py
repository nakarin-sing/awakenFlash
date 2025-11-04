#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
awakenFlash_benchmark.py
Real-World Benchmark for AwakenFlash Adaptive SRLS Model
"""

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# 🔧 Fix import path for GitHub Actions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import from awakenFlash core
from awakenFlash import AdaptiveSRLS

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

os.makedirs("benchmark_results", exist_ok=True)

print("Loading dataset (this may take a few seconds)...")

# === Synthetic streaming dataset generator ===
def load_synthetic_dataset(n_chunks=7, n_samples=2000, n_features=20, random_state=42):
    np.random.seed(random_state)
    data_chunks = []
    for i in range(n_chunks):
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=10,
            n_redundant=5,
            n_classes=7,
            random_state=random_state + i
        )
        data_chunks.append((X, y))
    return data_chunks


# === Scenario 4: Adaptive Streaming Learning ===
def scenario4_adaptive():
    print("\n===== Scenario 4: Adaptive Streaming Learning =====\n")

    data_stream = load_synthetic_dataset()
    acc_records = {"chunk": [], "SGD": [], "A-SRLS": [], "XGB": []}
    time_records = {"SGD": [], "A-SRLS": [], "XGB": []}

    # Initialize models
    sgd = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.01, random_state=42)
    asrls = AdaptiveSRLS(input_dim=20, alpha=0.05)

    if XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=30,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=7,
            verbosity=0
        )
    else:
        xgb = None

    classes = np.arange(7)

    # Process each data chunk sequentially
    for i, (X, y) in enumerate(data_stream, start=1):
        print(f"\n===== Processing Chunk {i:02d} =====")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # SGD
        t0 = time.time()
        sgd.partial_fit(X_train, y_train, classes=classes)
        y_pred_sgd = sgd.predict(X_test)
        acc_sgd = accuracy_score(y_test, y_pred_sgd)
        time_records["SGD"].append(time.time() - t0)

        # Adaptive SRLS
        t0 = time.time()
        asrls.partial_fit(X_train, y_train)
        y_pred_asrls = asrls.predict(X_test)
        acc_asrls = accuracy_score(y_test, y_pred_asrls)
        time_records["A-SRLS"].append(time.time() - t0)

        # XGBoost (retrain cumulative)
        if xgb is not None:
            t0 = time.time()
            if i == 1:
                X_train_full, y_train_full = X_train, y_train
            else:
                X_train_full = np.vstack([X_train_full, X_train])
                y_train_full = np.hstack([y_train_full, y_train])

            # ⚡ Safe class normalization
            y_train_full = np.clip(y_train_full, 0, 6)
            xgb.fit(X_train_full, y_train_full)
            y_pred_xgb = xgb.predict(X_test)
            acc_xgb = accuracy_score(y_test, y_pred_xgb)
            time_records["XGB"].append(time.time() - t0)
        else:
            acc_xgb = np.nan
            time_records["XGB"].append(0)

        acc_records["chunk"].append(i)
        acc_records["SGD"].append(acc_sgd)
        acc_records["A-SRLS"].append(acc_asrls)
        acc_records["XGB"].append(acc_xgb)

        print(f"SGD:   acc={acc_sgd:.3f}, time={time_records['SGD'][-1]:.3f}s")
        print(f"A-SRLS: acc={acc_asrls:.3f}, time={time_records['A-SRLS'][-1]:.3f}s")
        if xgb is not None:
            print(f"XGB:   acc={acc_xgb:.3f}, time={time_records['XGB'][-1]:.3f}s")

    df = pd.DataFrame(acc_records)
    df.to_csv("benchmark_results/scenario4_adaptive.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df["chunk"], df["SGD"], "-o", label="SGD")
    plt.plot(df["chunk"], df["A-SRLS"], "-o", label="Adaptive SRLS")
    if xgb is not None:
        plt.plot(df["chunk"], df["XGB"], "-o", label="XGBoost")
    plt.title("Adaptive Streaming Learning Benchmark")
    plt.xlabel("Chunk")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("benchmark_results/scenario4_adaptive.png")
    print("\n✅ Results saved in benchmark_results/scenario4_adaptive.*")


if __name__ == "__main__":
    scenario4_adaptive()
