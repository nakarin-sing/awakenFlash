#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 17 NON ONESTEP ŚŪNYATĀ
Streaming + Closed-Form + Random Fourier Features
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
# 17 NON ONESTEP ŚŪNYATĀ (RFF + Incremental)
# ========================================
class OneStepSunyata17Non:
    def __init__(self, D=500, C=100.0, gamma=0.1):
        self.D = D          # RFF dimension
        self.C = C          # Regularization
        self.gamma = gamma
        self.W = None       # Random projection matrix
        self.alpha = None   # Dual weights
        self.P_inv = None   # Inverse covariance (for Sherman-Morrison)
        self.classes_ = None
        self.n_samples = 0

    def _rff_features(self, X):
        """Random Fourier Features: phi(X) = [cos(WX), sin(WX)]"""
        if self.W is None:
            n_features = X.shape[1]
            self.W = np.random.normal(0, np.sqrt(self.gamma), (self.D // 2, n_features))
        proj = X @ self.W.T
        return np.hstack([np.cos(proj), np.sin(proj)]) * np.sqrt(2.0 / self.D)

    def _one_hot(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
        y_int = np.searchsorted(self.classes_, y)
        return np.eye(len(self.classes_))[y_int]

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes

        phi = self._rff_features(X)  # (n, D)
        y_hot = self._one_hot(y)     # (n, n_classes)

        n = X.shape[0]
        n_classes = len(self.classes_)

        # First batch
        if self.alpha is None:
            self.P_inv = np.eye(self.D) / self.C
            self.alpha = np.zeros((self.D, n_classes))
            self.n_samples = 0

        # Incremental update using Sherman-Morrison
        for i in range(n):
            phi_i = phi[i:i+1]  # (1, D)
            y_i = y_hot[i:i+1]  # (1, n_classes)

            # P_inv @ phi_i.T
            P_phi = self.P_inv @ phi_i.T  # (D, 1)

            # phi_i @ P_phi
            denom = 1 + phi_i @ P_phi  # scalar
            if denom == 0:
                continue

            # Update P_inv
            self.P_inv -= (P_phi @ P_phi.T) / denom

            # Update alpha
            self.alpha += P_phi @ (y_i - phi_i @ self.alpha).T

        self.n_samples += n
        return self

    def predict(self, X):
        if self.alpha is None:
            return np.zeros(len(X), dtype=int)

        phi = self._rff_features(X)
        scores = phi @ self.alpha  # (n_test, n_classes)
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
# 17 NON BENCHMARK
# ========================================
def scenario_17non(chunks, all_classes):
    print("\n" + "="*80)
    print("17 NON ONESTEP ŚŪNYATĀ SCENARIO")
    print("="*80)

    sunyata = OneStepSunyata17Non(D=600, C=100.0, gamma=0.05)
    results = []

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        print(f"Chunk {chunk_id:02d}/{len(chunks)}")

        # ONESTEP ŚŪNYATĀ
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
    print("17 NON FINAL RESULTS")
    print("="*80)
    s_acc = df['sunyata_acc'].mean()
    x_acc = df['xgb_acc'].mean()
    s_time = df['sunyata_time'].mean()
    x_time = df['xgb_time'].mean()

    print(f"ŚŪNYATĀ : Acc={s_acc:.4f} | Time={s_time:.4f}s")
    print(f"XGB     : Acc={x_acc:.4f} | Time={x_time:.4f}s")

    print("\n17 NON INSIGHT:")
    if s_acc >= x_acc and s_time < x_time:
        print(f"   ONESTEP ŚŪNYATĀ BEATS XGBoost")
        print(f"   WHILE BEING {x_time/s_time:.1f}x FASTER")
        print(f"   TRUE ONESTEP + STREAMING ACHIEVED.")
        print(f"   17 NON ACHIEVED. ABSOLUTE NIRVANA.")
    else:
        print(f"   Still in samsara.")

    return df


# ========================================
# MAIN
# ========================================
def main():
    print("="*80)
    print("17 NON awakenFlash ONESTEP ŚŪNYATĀ")
    print("="*80)

    chunks, all_classes = load_data()
    results = scenario_17non(chunks, all_classes)

    os.makedirs('benchmark_results', exist_ok=True)
    results.to_csv('benchmark_results/17non_onestep_results.csv', index=False)

    print("\n17 Non OneStep ŚŪNYATĀ complete.")


if __name__ == "__main__":
    main()
