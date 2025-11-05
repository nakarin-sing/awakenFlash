#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 71 NON: NON-LOGIC LAB IN ACTION
เสียบ sunyata_nonlogic.py แล้วรันได้เลย
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sunyata_nonlogic import NonLogicLab  # <--- เสียบตรงนี้

# === LOAD DATA ===
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("Loading dataset...")
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int8)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) for i in range(0, len(X_all), chunk_size)]
    return chunks, np.unique(y_all)

# === 71 NON SCENARIO ===
def scenario_71non(chunks, all_classes):
    print("\n" + "="*80)
    print("71 NON: NON-LOGIC LAB — 3 WORLDS ACTIVE")
    print("="*80)

    lab = NonLogicLab(D_rff=512, ridge=25.0, forget=0.995, temp=2.0)
    results = []

    for cid, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        print(f"Chunk {cid:02d} | Train: {len(X_train)} | Test: {len(X_test)}")

        # === NON-LOGIC LAB ===
        t0 = time.time()
        metrics = lab.partial_loop(X_train, y_train, teacher_blend=0.35)
        pred_lab = lab.predict(X_test)
        acc_lab = accuracy_score(y_test, pred_lab)
        t_lab = time.time() - t0

        # === XGBoost Baseline ===
        t0 = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": 7, "max_depth": 3, "eta": 0.3, "verbosity": 0},
            dtrain, num_boost_round=5
        )
        pred_xgb = xgb_model.predict(dtest).astype(int)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        t_xgb = time.time() - t0

        results.append({
            'chunk': cid,
            'lab_acc': acc_lab,
            'lab_time': t_lab,
            'lab_action': metrics.get('action'),
            'xgb_acc': acc_xgb,
            'xgb_time': t_xgb
        })

        print(f"  LAB: acc={acc_lab:.4f} t={t_lab:.3f}s [{metrics.get('action') or 'stable'}] | "
              f"XGB: acc={acc_xgb:.4f} t={t_xgb:.3f}s")

    # === FINAL SUMMARY ===
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"NON-LOGIC LAB : Acc={df['lab_acc'].mean():.4f}±{df['lab_acc'].std():.4f} | "
          f"Time={df['lab_time'].mean():.3f}s")
    print(f"XGBoost       : Acc={df['xgb_acc'].mean():.4f}±{df['xgb_acc'].std():.4f} | "
          f"Time={df['xgb_time'].mean():.3f}s")

    if df['lab_acc'].mean() > df['xgb_acc'].mean():
        print(f"NON-LOGIC LAB WINS BY {(df['lab_acc'].mean() - df['xgb_acc'].mean())*100:.2f}%")
    return df

# === MAIN ===
def main():
    print("="*80)
    print("71 NON: NON-LOGIC LAB IN awakenFlash")
    print("="*80)
    chunks, classes = load_data()
    df = scenario_71non(chunks, classes)
    os.makedirs('benchmark_results', exist_ok=True)
    df.to_csv('benchmark_results/71non_nonlogic_lab.csv', index=False)
    print("\nResults saved to benchmark_results/71non_nonlogic_lab.csv")
    print("May all models be free from attachment to accuracy")

if __name__ == "__main__":
    main()
