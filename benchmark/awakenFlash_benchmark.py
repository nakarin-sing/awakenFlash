#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py
FAIR real-world streaming benchmark for AwakenFlash
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb


def benchmark():
    print("🚀 Starting FAIR benchmark...")

    # === 1. Load dataset ===
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"🔗 Loading dataset from: {url}")

    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1  # Convert to 0–6

    # Normalize features (critical for SGD!)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    all_classes = np.unique(y_all)

    # Split into chunks
    chunk_size = 10000
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, len(X_all), chunk_size)]

    # === 2. Init models ===
    # Online learning models (optimized for streaming)
    sgd = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",  # Adaptive learning rate
        max_iter=5,               # More iterations per chunk
        warm_start=True,
        random_state=42
    )
    
    pa = PassiveAggressiveClassifier(
        C=0.01,                   # Regularization
        max_iter=5,
        warm_start=True,
        random_state=42
    )

    # Batch models (for comparison)
    xgb_model = None
    xgb_all_X, xgb_all_y = [], []  # Store all data for XGBoost

    first_sgd = True
    first_pa = True

    os.makedirs("benchmark_results", exist_ok=True)

    results = []

    # === 3. Stream loop ===
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks[:10], start=1):
        print(f"\n===== Processing Chunk {chunk_id:02d} =====")

        # Split chunk into train/test (80/20)
        split_idx = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split_idx], X_chunk[split_idx:]
        y_train, y_test = y_chunk[:split_idx], y_chunk[split_idx:]

        # ===== Online Learning: SGD =====
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X_train, y_train, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train)
        
        sgd_pred = sgd.predict(X_test)
        sgd_acc = accuracy_score(y_test, sgd_pred)
        sgd_time = time.time() - start
        print(f"SGD (Online):  acc={sgd_acc:.4f}, time={sgd_time:.4f}s")

        # ===== Online Learning: Passive-Aggressive =====
        start = time.time()
        if first_pa:
            pa.partial_fit(X_train, y_train, classes=all_classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train)
        
        pa_pred = pa.predict(X_test)
        pa_acc = accuracy_score(y_test, pa_pred)
        pa_time = time.time() - start
        print(f"PA (Online):   acc={pa_acc:.4f}, time={pa_time:.4f}s")

        # ===== Batch Learning: XGBoost =====
        # XGBoost needs to retrain on ALL historical data (fair comparison)
        start = time.time()
        xgb_all_X.append(X_train)
        xgb_all_y.append(y_train)
        
        X_xgb_train = np.vstack(xgb_all_X)
        y_xgb_train = np.concatenate(xgb_all_y)
        
        dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train with limited trees (fair capacity)
        xgb_model = xgb.train(
            {
                "objective": "multi:softmax",
                "num_class": 7,
                "max_depth": 4,           # Limited depth
                "eta": 0.3,               # Learning rate
                "verbosity": 0
            },
            dtrain,
            num_boost_round=10,           # Fixed number of trees
        )
        
        xgb_pred = xgb_model.predict(dtest)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_time = time.time() - start
        print(f"XGB (Batch):   acc={xgb_acc:.4f}, time={xgb_time:.4f}s")

        # Store results
        results.append({
            'chunk': chunk_id,
            'sgd_acc': sgd_acc,
            'sgd_time': sgd_time,
            'pa_acc': pa_acc,
            'pa_time': pa_time,
            'xgb_acc': xgb_acc,
            'xgb_time': xgb_time,
        })

    # === 4. Summary ===
    print("\n" + "="*60)
    print("📊 FINAL RESULTS (Average over all chunks)")
    print("="*60)
    
    df_results = pd.DataFrame(results)
    
    print(f"\nSGD (Online Learning):")
    print(f"  Avg Accuracy: {df_results['sgd_acc'].mean():.4f}")
    print(f"  Avg Time:     {df_results['sgd_time'].mean():.4f}s")
    
    print(f"\nPassive-Aggressive (Online Learning):")
    print(f"  Avg Accuracy: {df_results['pa_acc'].mean():.4f}")
    print(f"  Avg Time:     {df_results['pa_time'].mean():.4f}s")
    
    print(f"\nXGBoost (Batch Learning):")
    print(f"  Avg Accuracy: {df_results['xgb_acc'].mean():.4f}")
    print(f"  Avg Time:     {df_results['xgb_time'].mean():.4f}s")
    
    print("\n💡 Key Insights:")
    print("  - Online learning (SGD/PA) updates incrementally")
    print("  - XGBoost retrains on ALL historical data (expensive!)")
    print("  - Fair comparison: same train/test split for all models")
    
    # Save results
    df_results.to_csv("benchmark_results/results.csv", index=False)
    print("\n✅ Results saved to benchmark_results/results.csv")


if __name__ == "__main__":
    benchmark()
