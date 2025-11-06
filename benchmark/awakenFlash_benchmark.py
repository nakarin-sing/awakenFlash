#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ULTIMATE BENCHMARK: OneStep + Nyström vs XGBoost (10k, 50k, 100k samples)
- Full data (no subsample)
- Nyström approximation (m=2000)
- Fair comparison: same preprocessing, single thread, CPU time
- Saves results to benchmark_results/
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import psutil
import gc
from datetime import datetime
import time

def cpu_time():
    p = psutil.Process(os.getpid())
    return p.cpu_times().user + p.cpu_times().system

# ========================================
# 1. OneStep + Nyström
# ========================================
class OneStepNystrom:
    def __init__(self, C=1.0, n_components=2000, gamma='scale', random_state=42):
        self.C = C
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.landmarks_ = None
        self.alpha_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = self.scaler.fit_transform(X).astype(np.float32)
        n = X.shape[0]
        m = min(self.n_components, n)
        
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)[:m]
        self.landmarks_ = X[idx]
        
        gamma = 1.0 / X.shape[1] if self.gamma == 'scale' else self.gamma
        K_nm = np.exp(-gamma * (
            (X**2).sum(1)[:, None] + 
            (self.landmarks_**2).sum(1)[None, :] - 
            2 * X @ self.landmarks_.T
        ))
        K_mm = np.exp(-gamma * (
            (self.landmarks_**2).sum(1)[:, None] + 
            (self.landmarks_**2).sum(1)[None, :] - 
            2 * self.landmarks_ @ self.landmarks_.T
        ))
        
        lambda_reg = self.C * np.trace(K_mm) / m
        K_reg = K_mm + lambda_reg * np.eye(m, dtype=np.float32)
        
        self.classes_ = np.unique(y)
        y_onehot = np.zeros((n, len(self.classes_)), dtype=np.float32)
        for i, c in enumerate(self.classes_):
            y_onehot[y == c, i] = 1.0
        
        beta = np.linalg.solve(K_reg, K_nm.T @ y_onehot)
        self.alpha_ = K_nm @ beta
        
        return self

    def predict(self, X):
        X = self.scaler.transform(X).astype(np.float32)
        gamma = 1.0 / X.shape[1] if self.gamma == 'scale' else self.gamma
        K_test = np.exp(-gamma * (
            (X**2).sum(1)[:, None] + 
            (self.landmarks_**2).sum(1)[None, :] - 
            2 * X @ self.landmarks_.T
        ))
        scores = K_test @ self.alpha_
        return self.classes_[np.argmax(scores, axis=1)]

# ========================================
# 2. XGBoost Wrapper
# ========================================
class XGBoostWrapper:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=1,
            tree_method='hist',
            verbosity=0
        )

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)

# ========================================
# 3. Save Results
# ========================================
def save_results(filename, content):
    os.makedirs('benchmark_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f'benchmark_results/{filename}', 'w', encoding='utf-8') as f:
        f.write(f"# {timestamp}\n\n{content}\n")
    print(f"Saved: benchmark_results/{filename}")

# ========================================
# 4. Benchmark Function
# ========================================
def run_benchmark(n_train):
    print(f"\n{'='*80}")
    print(f"DATASET: {n_train:,} SAMPLES")
    print(f"{'='*80}")

    X, y = make_classification(
        n_samples=n_train + 20000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=20000, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    reps = 10

    # --- OneStep Nyström ---
    print(f"Training OneStep + Nyström (m=2000) {reps}x...")
    cpu_times = []
    accs = []
    for _ in range(reps):
        start = cpu_time()
        model = OneStepNystrom(C=1.0, n_components=2000, random_state=42)
        model.fit(X_train, y_train)
        cpu_times.append(cpu_time() - start)
        pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, pred))
    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    acc_mean = np.mean(accs)
    print(f"OneStep: {cpu_mean:.3f} ± {cpu_std:.3f}s | Acc: {acc_mean:.4f}")

    # --- XGBoost ---
    print(f"Training XGBoost {reps}x...")
    cpu_times = []
    accs = []
    for _ in range(reps):
        start = cpu_time()
        model = XGBoostWrapper()
        model.fit(X_train, y_train)
        cpu_times.append(cpu_time() - start)
        pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, pred))
    xgb_cpu_mean = np.mean(cpu_times)
    xgb_cpu_std = np.std(cpu_times)
    xgb_acc_mean = np.mean(accs)
    print(f"XGBoost: {xgb_cpu_mean:.3f} ± {xgb_cpu_std:.3f}s | Acc: {xgb_acc_mean:.4f}")

    # --- Summary ---
    speedup = xgb_cpu_mean / cpu_mean
    acc_diff = acc_mean - xgb_acc_mean
    winner_speed = "OneStep" if cpu_mean < xgb_cpu_mean else "XGBoost"
    winner_acc = "OneStep" if acc_mean > xgb_acc_mean else "XGBoost" if xgb_acc_mean > acc_mean else "TIE"

    print(f"\nSPEEDUP: OneStep {speedup:.2f}x faster")
    print(f"ACC DIFF: OneStep {'+' if acc_diff >= 0 else ''}{acc_diff:.4f}")
    print(f"WINNER: Speed={winner_speed}, Acc={winner_acc}")

    # Save
    content = f"""NYSTROM BENCHMARK - {n_train:,} SAMPLES
OneStep: {cpu_mean:.3f}±{cpu_std:.3f}s, Acc: {acc_mean:.4f}
XGBoost: {xgb_cpu_mean:.3f}±{xgb_cpu_std:.3f}s, Acc: {xgb_acc_mean:.4f}
Speedup: {speedup:.2f}x | Acc Diff: {acc_diff:.4f}
Winner: Speed={winner_speed}, Acc={winner_acc}"""
    save_results(f"nystrom_{n_train//1000}k.txt", content)

    return {
        'n': n_train,
        'onestep_cpu': cpu_mean,
        'onestep_acc': acc_mean,
        'xgboost_cpu': xgb_cpu_mean,
        'xgboost_acc': xgb_acc_mean,
        'speedup': speedup,
        'acc_diff': acc_diff
    }

# ========================================
# 5. Main
# ========================================
if __name__ == "__main__":
    os.system("mkdir -p benchmark_results")
    results = []
    
    for n in [10000, 50000, 100000]:
        results.append(run_benchmark(n))
        gc.collect()

    # Final Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['n']//1000:3d}k | OneStep: {r['onestep_cpu']:5.2f}s ({r['onestep_acc']:.4f}) | "
              f"XGBoost: {r['xgboost_cpu']:5.2f}s ({r['xgboost_acc']:.4f}) | "
              f"Speedup: {r['speedup']:4.1f}x | Acc: {'+' if r['acc_diff'] >= 0 else ''}{r['acc_diff']:.4f}")

    summary = "\n".join([
        f"NYSTROM vs XGBOOST SUMMARY",
        f"{'Size':>6} | {'OneStep CPU':>12} | {'OneStep Acc':>12} | {'XGBoost CPU':>12} | {'XGBoost Acc':>12} | {'Speedup':>8} | {'Acc Diff':>8}",
        f"{'-'*80}"
    ] + [
        f"{r['n']//1000:6d}k | {r['onestep_cpu']:10.3f}s | {r['onestep_acc']:10.4f} | "
        f"{r['xgboost_cpu']:10.3f}s | {r['xgboost_acc']:10.4f} | {r['speedup']:6.1f}x | "
        f"{'+' if r['acc_diff'] >= 0 else ''}{r['acc_diff']:6.4f}"
        for r in results
    ])
    save_results("nystrom_summary.txt", summary)
