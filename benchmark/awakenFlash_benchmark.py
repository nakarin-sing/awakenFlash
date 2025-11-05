#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 47 NON: RLS RESURRECTION
O(D²) Update | Cold Start Fix | Pure Imagination
"""

import os
import time
import numpy as np
import pandas as pd
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# === VECTORIZED ABSOLUTE NON ===
class VectorizedAbsoluteNon:
    def __init__(self, n=1, α=0.7, β=0.6, γ=0.95, δ=0.9):
        self.n = n
        self.α = α
        self.β = β
        self.γ = γ
        self.δ = δ
        self.log2 = np.log(2.0)
        self.pi = np.pi
        self.sqrt_pi = np.sqrt(np.pi)
        self.sign_n = 1.0 if n % 2 == 0 else -1.0

    def transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        x = np.clip(x, -5.0, 5.0)
        abs_diff = np.abs(x - 0.5)
        sym = self.α * np.exp(-self.n * self.log2 * abs_diff)
        flow = (1 - self.α) * x * np.exp(-self.n * self.log2)
        enlight = self.β * (np.sin(self.pi * x) + 0.5 * np.cos(2 * self.pi * x))
        compassion = self.δ * (1 - abs_diff) / np.sqrt(1 + x**2)
        linear = 0.5 + self.sign_n * (x - 0.5) * np.exp(-(self.n - 1) * self.log2)
        non_core = sym + flow + linear * 1e-12
        full = non_core + (1 - self.β) * enlight + (1 - self.δ) * compassion
        meta = self.γ * np.exp(-x**2) / self.sqrt_pi * np.cos(2 * self.pi * x)
        return self.γ * meta + (1 - self.γ) * full

_abs_non = VectorizedAbsoluteNon(n=1)

# === 47 NON: RLS RESURRECTION ===
class SunyataV15_RLS:
    def __init__(self, D_init=1024, C=200.0, forgetting=0.99, seed=42, buffer_size=2000):
        self.D = int(D_init)
        self.C = float(C)
        self.forgetting = float(forgetting)
        self.rng = np.random.default_rng(seed)
        self.W = None
        self.alpha = None          # (D, K)
        self.P = None              # (D, D) — inverse covariance
        self.classes_ = None
        self.class_to_idx = {}
        self.buffer_size = int(buffer_size)
        self.buffer = collections.deque(maxlen=self.buffer_size)  # (phi_non, y_onehot)
        self.eps = 1e-6
        self.initialized = False

    def _rff_features(self, X):
        n_features = X.shape[1]
        target_rows = self.D // 2
        if self.W is None or self.W.shape[1] != n_features:
            scale = 1.0 / np.sqrt(n_features)
            self.W = self.rng.normal(0, scale, (target_rows, n_features)).astype(np.float32)
        X32 = X.astype(np.float32)
        proj = X32 @ self.W.T
        phi = np.hstack([np.cos(proj), np.sin(proj)]) * np.sqrt(2.0 / self.D)
        return phi

    def _augment_features(self, phi):
        n, D = phi.shape
        m = min(64, D//4)
        if m > 0:
            inter = (phi[:, :m] * phi[:, m:2*m])
            phi_aug = np.hstack([phi, inter])
        else:
            phi_aug = phi
        mu = phi_aug.mean(axis=0, keepdims=True)
        std = phi_aug.std(axis=0, keepdims=True) + self.eps
        return (phi_aug - mu) / std

    def _encode_labels(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        return np.array([self.class_to_idx.get(val, 0) for val in y], dtype=np.int32)

    def _initialize_with_full_solve(self, phi_non, y_onehot):
        n, D = phi_non.shape
        K = y_onehot.shape[1]
        PhiT_Phi = (phi_non.T @ phi_non) / n
        PhiT_y = (phi_non.T @ y_onehot) / n
        H = PhiT_Phi + np.eye(D) * (self.C / n)
        self.alpha = np.linalg.solve(H, PhiT_y)
        self.P = np.linalg.inv(H)  # Initialize P
        self.initialized = True

    def _rls_update(self, phi, y_onehot):
        # RLS Update Rule: O(D²)
        for i in range(phi.shape[0]):
            x = phi[i:i+1]  # (1, D)
            y = y_onehot[i:i+1]  # (1, K)
            # P @ x.T: (D, D) @ (D, 1) → (D, 1)
            Px = self.P @ x.T
            # gain = Px / (1 + x @ Px)
            denom = 1.0 + float(x @ Px)
            if denom < self.eps:
                continue
            k = Px / denom  # (D, 1)
            # alpha update
            self.alpha += k @ (y - x @ self.alpha).T
            # P update: P = P - k @ (x @ P)
            self.P -= np.outer(k, x @ self.P)
            # Forgetting
            self.P *= self.forgetting
            # Regularization
            self.P += (1 - self.forgetting) * np.eye(self.P.shape[0]) * (self.C / self.D)

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        X32 = X.astype(np.float32)
        y_idx = self._encode_labels(y)
        phi = self._rff_features(X32)
        phi_non = _abs_non.transform(phi)
        phi_non = self._augment_features(phi_non)
        n, D_aug = phi_non.shape
        K = len(self.classes_)

        # Self-distillation: use current prediction
        if self.alpha is not None:
            scores = phi_non @ self.alpha
            probs = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            confidence = np.mean(np.max(probs, axis=1))
            distill_alpha = min(0.7, 0.1 + 0.6 * confidence)
        else:
            probs = np.eye(K)[y_idx]
            distill_alpha = 0.0

        hard = np.zeros((n, K), dtype=np.float32)
        hard[np.arange(n), y_idx] = 1.0
        y_onehot = (1.0 - distill_alpha) * hard + distill_alpha * probs

        # === COLD START: Full Solve on First Chunk ===
        if not self.initialized and n >= 512:
            self._initialize_with_full_solve(phi_non, y_onehot)
            # Fill buffer with current data
            for i in range(min(n, self.buffer_size)):
                self.buffer.append((phi_non[i].copy(), y_onehot[i].copy()))
            return self

        # === RLS UPDATE ON BUFFER + NEW DATA ===
        # Add new data to buffer
        for i in range(n):
            self.buffer.append((phi_non[i].copy(), y_onehot[i].copy()))

        # Sample from buffer
        buf_n = len(self.buffer)
        if buf_n == 0:
            return self

        sample_n = min(512, buf_n)
        idxs = self.rng.choice(buf_n, sample_n, replace=False)
        phi_buf = np.stack([self.buffer[i][0] for i in idxs])
        y_buf = np.stack([self.buffer[i][1] for i in idxs])

        # RLS Update
        self._rls_update(phi_buf, y_buf)

        return self

    def predict(self, X):
        if self.alpha is None:
            return np.full(len(X), self.classes_[0] if self.classes_ is not None else 0, dtype=np.int32)
        phi = self._rff_features(X.astype(np.float32))
        phi_non = _abs_non.transform(phi)
        phi_non = self._augment_features(phi_non)
        scores = phi_non @ self.alpha
        return self.classes_[np.argmax(scores, axis=1)]

# === BENCHMARK ===
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int8)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)

def scenario_47non(chunks, all_classes):
    print("\n" + "="*80)
    print("47 NON: RLS RESURRECTION — O(D²) SPEED DEMON")
    print("="*80)
    sunyata = SunyataV15_RLS(D_init=1024, C=200.0, forgetting=0.99, buffer_size=2000)
    results = []

    for cid, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        print(f"Chunk {cid:02d}/{len(chunks)} | Init={sunyata.initialized} | Buffer={len(sunyata.buffer)}")

        t0 = time.time()
        sunyata.partial_fit(X_train, y_train, classes=all_classes)
        pred_s = sunyata.predict(X_test)
        acc_s = accuracy_score(y_test, pred_s)
        t_s = time.time() - t0

        t0 = time.time()
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train({"objective": "multi:softmax", "num_class": 7, "max_depth": 3, "eta": 0.3}, dtrain, num_boost_round=5)
        pred_x = xgb_model.predict(dtest).astype(int)
        acc_x = accuracy_score(y_test, pred_x)
        t_x = time.time() - t0

        results.append({'chunk': cid, 's_acc': acc_s, 's_time': t_s, 'x_acc': acc_x, 'x_time': t_x})
        print(f"  RLS ŚŪNYATĀ: acc={acc_s:.3f} t={t_s:.3f}s  |  XGB: acc={acc_x:.3f} t={t_x:.3f}s")

    df = pd.DataFrame(results)
    print("\nRLS RESURRECTION ACHIEVED")
    s_acc = df['s_acc'].mean()
    x_acc = df['x_acc'].mean()
    s_time = df['s_time'].mean()
    x_time = df['x_time'].mean()
    print(f"RLS ŚŪNYATĀ: {s_acc:.4f} | {s_time:.3f}s")
    print(f"XGB:         {x_acc:.4f} | {x_time:.3f}s")
    if s_time < 0.05:
        print("=> SPEED DEMON: < 0.05s per chunk!")
    if s_acc > x_acc:
        print("=> ACCURACY ALSO WINS.")
    return df

def main():
    print("="*80)
    print("47 NON: RLS RESURRECTION IN awakenFlash")
    print("="*80)
    chunks, all_classes = load_data()
    df = scenario_47non(chunks, all_classes)
    os.makedirs('benchmark_results', exist_ok=True)
    df.to_csv('benchmark_results/47non_rls_resurrection.csv', index=False)

if __name__ == "__main__":
    main()
