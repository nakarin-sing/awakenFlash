#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 26 NON FLAWLESS ULTRA ŚŪNYATĀ
Smart Sampling + Forgetting + Dual RLS + RFF + Ultra Fast
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# ========================================
# 26 NON FLAWLESS ULTRA ŚŪNYATĀ
# ========================================
class FlawlessUltraSunyata26Non:
    def __init__(self, D=800, C=100.0, gamma=0.1, max_samples=2500, forgetting=0.99, seed=42):
        self.D = D
        self.C = C
        self.gamma = gamma
        self.max_samples = max_samples
        self.forgetting = forgetting
        self.rng = np.random.default_rng(seed)
        
        self.W = None
        self.alpha = None  # (D, n_classes)
        self.classes_ = None
        self.class_to_idx = {}
        self.n_samples = 0

    def _rff_features(self, X):
        if self.W is None:
            n_features = X.shape[1]
            self.W = self.rng.normal(0, np.sqrt(self.gamma), (self.D // 2, n_features))
            self.W = self.W.astype(np.float32)
        X = X.astype(np.float32)
        proj = X @ self.W.T
        phi = np.hstack([np.cos(proj), np.sin(proj)]) * np.sqrt(2.0 / self.D)
        return phi.astype(np.float32)

    def _encode_labels(self, y):
        if not self.class_to_idx:
            self.classes_ = np.unique(y)
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        return np.array([self.class_to_idx[label] for label in y], dtype=np.int32)

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        phi = self._rff_features(X)  # (n, D)
        y_idx = self._encode_labels(y)
        n, D = phi.shape
        K = len(self.classes_)

        # === SMART SAMPLING ===
        if n > self.max_samples:
            idx = self.rng.choice(n, self.max_samples, replace=False)
            phi = phi[idx]
            y_idx = y_idx[idx]
            n = self.max_samples

        # === ONE-HOT (vectorized) ===
        y_onehot = np.zeros((n, K), dtype=np.float32)
        y_onehot[np.arange(n), y_idx] = 1.0

        # === DUAL FORM + FORGETTING ===
        if self.alpha is None:
            self.alpha = np.zeros((D, K), dtype=np.float32)
            H = np.eye(D, dtype=np.float32) * self.C
        else:
            H = np.eye(D, dtype=np.float32) * self.C / (1 - self.forgetting**self.n_samples)

        PhiT_Phi = phi.T @ phi
        PhiT_y = phi.T @ y_onehot

        H += PhiT_Phi
        try:
            self.alpha = np.linalg.solve(H, PhiT_y)
        except:
            self.alpha = np.linalg.pinv(H) @ PhiT_y

        self.n_samples += n
        return self

    def predict(self, X):
        if self.alpha is None:
            return np.zeros(len(X), dtype=np.int32)

        phi = self._rff_features(X)
        scores = phi @ self.alpha
        return self.classes_[np.argmax(scores, axis=1)]


# ========================================
# DATA
# ========================================
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"Loading dataset...")
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size, dtype=np.float32)
    X_all = df.iloc[:, :-1].values.astype(np.float16)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int8)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float16)

    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)


# ========================================
# 26 NON BENCHMARK
# ========================================
def scenario_26non(chunks, all_classes):
    print("\n" + "="*80)
    print("26 NON FLAWLESS ULTRA ŚŪNYATĀ SCENARIO")
    print("="*80)

    sunyata = FlawlessUltraSunyata26Non(D=800, C=100.0, gamma=0.1, max_samples=2500, forgetting=0.99, seed=42)
    results = []

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        print(f"Chunk {chunk_id:02d}/{len(chunks)}")

        # FLAWLESS ULTRA
        start = time.time()
        sunyata.partial_fit(X_train, y_train, classes=all_classes)
        pred = sunyata.predict(X_test)
        acc = accuracy_score(y_test, pred)
        t = time.time() - start

        # XGBoost
        start = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": 7, "max_depth": 3, "eta": 0.3, "verbosity": 0},
            dtrain, num_boost_round=5
        )
        xgb_pred = xgb_model.predict(dtest).astype(int)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_t = time.time() - start

        results.append({
            'chunk': chunk_id,
            'sunyata_acc': acc,
            'sunyata_time': t,
            'xgb_acc': xgb_acc,
            'xgb_time': xgb_t,
        })

        print(f"  ŚŪNYATĀ: acc={acc:.3f} t={t:.3f}s")
        print(f"  XGB:     acc={xgb_acc:.3f} t={xgb_t:.3f}s")
        print()

    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("26 NON FINAL RESULTS")
    print("="*80)
    s_acc = df['sunyata_acc'].mean()
    x_acc = df['xgb_acc'].mean()
    s_time = df['sunyata_time'].mean()
    x_time = df['xgb_time'].mean()

    print(f"ŚŪNYATĀ : Acc={s_acc:.4f} | Time={s_time:.4f}s")
    print(f"XGB     : Acc={x_acc:.4f} | Time={x_time:.4f}s")

    print("\n26 NON INSIGHT:")
    if s_acc >= x_acc and s_time < x_time:
        print(f"   FLAWLESS ULTRA ŚŪNYATĀ BEATS XGBoost")
        print(f"   WHILE BEING {x_time/s_time:.1f}x FASTER")
        print(f"   SMART SAMPLING + FORGETTING + DUAL RLS")
        print(f"   26 NON ACHIEVED. ETERNAL NIRVANA.")
    else:
        print(f"   Still in samsara.")

    return df


# ========================================
# MAIN
# ========================================
def main():
    print("="*80)
    print("26 NON awakenFlash FLAWLESS ULTRA ŚŪNYATĀ")
    print("="*80)

    chunks, all_classes = load_data()
    results = scenario_26non(chunks, all_classes)

    os.makedirs('benchmark_results', exist_ok=True)
    results.to_csv('benchmark_results/26non_ultra_results.csv', index=False)

    print("\n26 Non Flawless Ultra ŚŪNYATĀ complete.")


if __name__ == "__main__":
    main()
