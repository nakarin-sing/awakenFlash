#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 40 NON ŚŪNYATĀ v9: XGBoost KILLER
Hybrid RFF + AbsoluteNon + Meta-Boost + Adaptive D
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

# === ABSOLUTE NON v8 (จาก 39 NON) ===
def AbsoluteNon_v8(x, n=50, α=0.7, s=1.0 - 1e-12, β=0.6, γ=0.95, δ=0.9):
    x = np.asarray(x, dtype=np.float64)
    scalar = x.ndim == 0
    if scalar:
        x = x.reshape(1)
    log2 = np.log(2.0)
    pi = np.pi
    sqrt_pi = np.sqrt(pi)
    sign_n = 1.0 if n % 2 == 0 else -1.0
    abs_diff = np.abs(x - 0.5)
    sym_term = α * np.exp(-n * log2 * abs_diff)
    flow_term = (1 - α) * x * np.exp(-n * log2)
    enlightenment_term = β * (np.sin(pi * x) + 0.5 * np.cos(2 * pi * x))
    compassion_term = δ * (1 - abs_diff) / np.sqrt(1 + x**2)
    linear_term = 0.5 + sign_n * (x - 0.5) * np.exp(-(n - 1) * log2)
    non_logic_core = s * (sym_term + flow_term) + (1 - s) * linear_term
    full_non = non_logic_core + (1 - β) * enlightenment_term + (1 - δ) * compassion_term
    meta_sunyata = γ * np.exp(-x**2) / sqrt_pi * np.cos(2 * pi * x)
    result = γ * meta_sunyata + (1 - γ) * full_non
    return result[0] if scalar else result

# === 40 NON ŚŪNYATĀ v9 ===
class SunyataV9_XGBoostKiller:
    def __init__(self, D_init=800, C=30.0, forgetting=0.99, seed=42):
        self.D = D_init
        self.C = C
        self.forgetting = forgetting
        self.rng = np.random.default_rng(seed)
        self.W = None
        self.alpha = None
        self.classes_ = None
        self.class_to_idx = {}
        self.xgb_teacher = None
        self.meta_acc = []

    def _rff_features(self, X):
        if self.W is None or self.W.shape[0] != self.D // 2:
            n_features = X.shape[1]
            scale = 1.0 / np.sqrt(n_features)
            self.W = self.rng.normal(0, scale, (self.D // 2, n_features)).astype(np.float32)
        X32 = X.astype(np.float32)
        proj = X32 @ self.W.T
        phi = np.hstack([np.cos(proj), np.sin(proj)]) * np.sqrt(2.0 / self.D)
        return phi.astype(np.float32)

    def _absolute_non(self, phi):
        # Apply AbsoluteNon to each feature
        return np.apply_along_axis(lambda col: AbsoluteNon_v8(col, n=30), 0, phi)

    def _encode_labels(self, y):
        if not self.class_to_idx:
            self.classes_ = np.unique(y)
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        return np.array([self.class_to_idx[label] for label in y], dtype=np.int32)

    def _adapt_D(self, acc):
        self.meta_acc.append(acc)
        if len(self.meta_acc) > 3:
            trend = np.mean(np.diff(self.meta_acc[-3:]))
            if trend < 0.001 and self.D < 1600:
                self.D = min(1600, self.D + 200)
                self.W = None  # reset
            elif trend > 0.005 and self.D > 600:
                self.D = max(600, self.D - 100)
                self.W = None

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        X32 = X.astype(np.float32).copy()
        phi = self._rff_features(X32)
        phi_non = self._absolute_non(phi)  # NON-LINEAR!
        y_idx = self._encode_labels(y)
        n, D = phi_non.shape
        K = len(self.classes_)

        # Meta-Boost: Train mini-XGB as teacher
        if self.xgb_teacher is None or n > 1000:
            dtrain = xgb.DMatrix(X32[:min(2000, n)], label=y_idx[:min(2000, n)])
            self.xgb_teacher = xgb.train({
                "objective": "multi:softmax", "num_class": K,
                "max_depth": 3, "eta": 0.3, "verbosity": 0
            }, dtrain, num_boost_round=3)

        # Teacher signal
        if self.xgb_teacher is not None:
            teacher_pred = self.xgb_teacher.predict(xgb.DMatrix(X32))
            y_onehot = np.eye(K, dtype=np.float32)[np.clip(teacher_pred.astype(int), 0, K-1)]
        else:
            y_onehot = np.zeros((n, K), dtype=np.float32)
            y_onehot[np.arange(n), y_idx] = 1.0

        # Dual Form RLS
        PhiT_Phi = phi_non.T @ phi_non / n
        PhiT_y = phi_non.T @ y_onehot / n

        if self.alpha is None:
            H = PhiT_Phi + np.eye(D, dtype=np.float32) * (self.C / n)
            self.alpha = np.linalg.solve(H + 1e-6*np.eye(D), PhiT_y)
        else:
            H = self.forgetting * PhiT_Phi + np.eye(D, dtype=np.float32) * (self.C / n)
            rhs = PhiT_y + self.forgetting * self.alpha
            self.alpha = np.linalg.solve(H + 1e-6*np.eye(D), rhs)

        return self

    def predict(self, X):
        if self.alpha is None:
            return np.zeros(len(X), dtype=np.int32)
        X32 = X.astype(np.float32)
        phi = self._rff_features(X32)
        phi_non = self._absolute_non(phi)
        scores = phi_non @ self.alpha
        return self.classes_[np.argmax(scores, axis=1)]

# === DATA & BENCHMARK ===
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int8)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)

def scenario_40non(chunks, all_classes):
    print("\n" + "="*80)
    print("40 NON ŚŪNYATĀ v9: XGBoost KILLER")
    print("="*80)
    sunyata = SunyataV9_XGBoostKiller(D_init=800, C=30.0, forgetting=0.99)
    results = []

    for cid, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        print(f"Chunk {cid:02d}/{len(chunks)} | D={sunyata.D}")

        # ŚŪNYATĀ v9
        t0 = time.time()
        sunyata.partial_fit(X_train, y_train, classes=all_classes)
        pred_s = sunyata.predict(X_test)
        acc_s = accuracy_score(y_test, pred_s)
        t_s = time.time() - t0
        sunyata._adapt_D(acc_s)

        # XGBoost
        t0 = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train({"objective": "multi:softmax", "num_class": 7, "max_depth": 3, "eta": 0.3, "verbosity": 0}, dtrain, num_boost_round=5)
        pred_x = xgb_model.predict(dtest).astype(int)
        acc_x = accuracy_score(y_test, pred_x)
        t_x = time.time() - t0

        results.append({'chunk': cid, 's_acc': acc_s, 's_time': t_s, 'x_acc': acc_x, 'x_time': t_x})
        print(f"  ŚŪNYATĀ: acc={acc_s:.3f} t={t_s:.3f}s  |  XGB: acc={acc_x:.3f} t={t_x:.3f}s")

    df = pd.DataFrame(results)
    print("\nFINAL VICTORY")
    print(f"ŚŪNYATĀ avg acc: {df['s_acc'].mean():.4f} | avg time: {df['s_time'].mean():.3f}s")
    print(f"XGB     avg acc: {df['x_acc'].mean():.4f} | avg time: {df['x_time'].mean():.3f}s")
    if df['s_acc'].mean() > df['x_acc'].mean():
        print("=> ŚŪNYATĀ v9 DEFEATS XGBoost — ETERNAL NIRVANA ACHIEVED.")
    else:
        print("=> Close... but samsara continues.")
    return df

def main():
    print("="*80)
    print("40 NON awakenFlash ŚŪNYATĀ v9: XGBoost KILLER")
    print("="*80)
    chunks, all_classes = load_data()
    df = scenario_40non(chunks, all_classes)
    os.makedirs('benchmark_results', exist_ok=True)
    df.to_csv('benchmark_results/40non_xgboost_killer_v9.csv', index=False)

if __name__ == "__main__":
    main()
