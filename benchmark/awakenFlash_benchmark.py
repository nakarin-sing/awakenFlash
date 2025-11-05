#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark_sunyata_v2.py
ŚŪNYATĀ v2 — Let Go of Emptiness (adaptive kernel + temporal RLS for streaming)
- Adaptive σ (kernel scale) per chunk
- Adaptive mean re-centering (per-chunk)
- Temporal decay (recency weighting) in RLS
- Dual-branch warm-start influence (non-logic state -> RLS bias)
Designed for streaming benchmarks vs XGBoost (fast, low latency)
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

# ---------------------------
# Utilities
# ---------------------------
def safe_inv(mat, eps=1e-6):
    # stable inverse via eigh for symmetric matrices
    try:
        return np.linalg.inv(mat)
    except Exception:
        return np.linalg.pinv(mat)

# ---------------------------
# Non-Logic warm meta-state (simple, fast)
# ---------------------------
class NonLogicState:
    def __init__(self, blend=0.2):
        self.state = None
        self.blend = blend  # how strongly warm-start affects RLS bias

    def consume(self, X):
        # signature: normalized mean projection
        sig = np.tanh(np.mean(X, axis=0))
        if self.state is None:
            self.state = sig
        else:
            self.state = (1 - self.blend) * self.state + self.blend * sig

    def bias_vector(self, D):
        # expand to D dims by simple random projection seeded from state
        if self.state is None:
            return None
        rng = np.random.default_rng(np.int64((np.abs(self.state).sum() * 1e6)) & 0xffffffff)
        v = rng.standard_normal(D).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        return v

# ---------------------------
# ŚŪNYATĀ v2 Hybrid Online Model
# ---------------------------
class SunyataV2:
    def __init__(self, D=512, ridge=1.0, decay=0.90, init_sigma=1.0):
        """
        D: random feature dimension (should be even)
        ridge: regularization coefficient
        decay: temporal forgetting factor in RLS (0<decay<=1)
        init_sigma: initial kernel scale
        """
        assert D % 2 == 0
        self.D = D
        self.ridge = ridge
        self.decay = decay
        self.sigma = float(init_sigma)
        self.W = None            # random projection (D/2 x feat)
        self.alpha = None        # (D x K) weight matrix for one-hot outputs
        self.P = None            # inverse covariance for RLS (D x D) or block diag
        self.classes_ = None
        self.class_to_idx = {}
        self.nl_state = NonLogicState(blend=0.25)
        self.global_mean = None
        self.n_seen = 0

    def _ensure_W(self, n_feat):
        if self.W is None:
            rng = np.random.default_rng(12345)
            # scale of W chosen as 1/sigma later (we adapt sigma by scaling proj)
            self.W = rng.normal(0, 1.0, size=(self.D // 2, n_feat)).astype(np.float32)

    def _rff(self, X):
        # Random Fourier Features with adaptive sigma
        # phi = sqrt(2/D) [cos(X W / sigma), sin(X W / sigma)]
        proj = (X.astype(np.float32) @ self.W.T) / (self.sigma + 1e-12)
        phi = np.hstack([np.cos(proj), np.sin(proj)]) * np.sqrt(2.0 / self.D)
        return phi.astype(np.float32)

    def _encode(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self.class_to_idx[yy] for yy in y], dtype=np.int32)

    def adapt_sigma(self, X):
        # set sigma as median pairwise projection scale proxy: use std of projected features
        # cheap proxy: take std of features along axis
        std_feat = np.std(X, axis=0).mean()
        # map to sigma with smoothing
        target = max(0.01, float(std_feat) + 0.1)
        # exponential smoothing
        self.sigma = 0.7 * self.sigma + 0.3 * target

    def recenter(self, X):
        # adaptive mean re-centering to remove chunk shift
        chunk_mean = np.mean(X, axis=0)
        if self.global_mean is None:
            self.global_mean = chunk_mean
        else:
            # moving average with weight proportional to number seen
            beta = 0.2
            self.global_mean = (1 - beta) * self.global_mean + beta * chunk_mean
        Xc = X - self.global_mean
        return Xc

    def partial_fit(self, X, y, classes=None):
        """
        Online partial fit using recursive least squares on random features.
        We incorporate:
         - temporal decay on P (inverse covariance)
         - warm-start bias from NonLogicState
        """
        n, feat = X.shape
        if classes is not None:
            self.classes_ = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        # update nonlogic state
        self.nl_state.consume(X)

        # adapt center and sigma
        Xc = self.recenter(X)
        self.adapt_sigma(Xc)

        # ensure W
        self._ensure_W(feat)
        Phi = self._rff(Xc)           # shape (n, D)
        y_idx = self._encode(y)
        K = len(self.classes_)

        # one-hot labels
        Y = np.zeros((n, K), dtype=np.float32)
        Y[np.arange(n), y_idx] = 1.0

        # initialize alpha and P if first time
        if self.alpha is None:
            self.alpha = np.zeros((self.D, K), dtype=np.float32)
            # initialize P as (1/ridge) * I
            self.P = (1.0 / self.ridge) * np.eye(self.D, dtype=np.float32)

            # if nonlogic provides bias vector, nudge alpha initial
            bias = self.nl_state.bias_vector(self.D)
            if bias is not None:
                # distribute bias across classes proportional to class prior of batch
                class_prior = Y.mean(axis=0, keepdims=True).astype(np.float32) + 1e-6
                self.alpha += np.outer(bias, class_prior.ravel()) * 0.1  # small warm push

        # RLS batch update with temporal decay
        # Process rows in mini-batches for stability
        chunk_size = max(64, min(512, n))
        for i0 in range(0, n, chunk_size):
            Xi = Phi[i0:i0+chunk_size]  # (m, D)
            Yi = Y[i0:i0+chunk_size]    # (m, K)
            # RLS update (matrix form):
            # P <- (1/decay) * (P - P Xi^T (I + Xi P Xi^T)^-1 Xi P)
            # alpha <- alpha + P Xi^T (Yi - Xi alpha)
            Pi = self.P
            # compute S = I + Xi Pi Xi^T
            S = Xi @ Pi @ Xi.T
            S[np.diag_indices_from(S)] += 1.0  # add identity
            S_inv = safe_inv(S)
            Kmat = Pi @ Xi.T @ S_inv  # (D, m)
            # predictive error
            err = Yi - Xi @ self.alpha  # (m, K)
            # alpha update
            self.alpha = self.alpha + Kmat @ err
            # P update with decay
            self.P = (1.0 / self.decay) * (Pi - Kmat @ Xi @ Pi)

        self.n_seen += n
        return self

    def predict(self, X):
        if self.alpha is None:
            return np.zeros(len(X), dtype=int)
        Xc = X - (self.global_mean if self.global_mean is not None else np.mean(X, axis=0))
        Phi = self._rff(Xc)
        scores = Phi @ self.alpha  # (n, K)
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


# ---------------------------
# Data loader (covtype trimmed for speed)
# ---------------------------
def load_data(n_chunks=8, chunk_size=5000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("Loading dataset (may take a moment)...")
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int32)
    # global scaling kept but we will recenter per chunk
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)

# ---------------------------
# Benchmark scenario (streaming)
# ---------------------------
def scenario_sunyata_v2(chunks, all_classes):
    print("\n" + "="*80)
    print("STREAMING: ŚŪNYATĀ v2 (adaptive) vs streaming-like XGBoost")
    print("="*80)
    model = SunyataV2(D=512, ridge=2.0, decay=0.92, init_sigma=0.8)
    results = []
    for cid, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        print(f"Chunk {cid:02d}/{len(chunks)} | train={len(X_train)} test={len(X_test)}")
        # SUNYATA
        t0 = time.time()
        model.partial_fit(X_train, y_train, classes=all_classes)
        pred_s = model.predict(X_test)
        t_s = time.time() - t0
        acc_s = accuracy_score(y_test, pred_s)
        # XGBoost (sliding-window single-chunk train)
        t0 = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train({"objective": "multi:softmax", "num_class": len(all_classes),
                               "max_depth": 4, "eta": 0.2, "verbosity": 0}, dtrain, num_boost_round=8)
        pred_x = xgb_model.predict(dtest).astype(int)
        t_x = time.time() - t0
        acc_x = accuracy_score(y_test, pred_x)
        print(f"  ŚŪNYATĀ: acc={acc_s:.3f} t={t_s:.3f}s  |  XGB: acc={acc_x:.3f} t={t_x:.3f}s")
        results.append({'chunk': cid, 's_acc': acc_s, 's_time': t_s, 'x_acc': acc_x, 'x_time': t_x})
    df = pd.DataFrame(results)
    print("\nFINAL")
    print("ŚŪNYATĀ avg acc: {:.4f} | avg time: {:.3f}s".format(df['s_acc'].mean(), df['s_time'].mean()))
    print("XGB     avg acc: {:.4f} | avg time: {:.3f}s".format(df['x_acc'].mean(), df['x_time'].mean()))
    if df['s_acc'].mean() >= df['x_acc'].mean():
        print("=> ŚŪNYATĀ v2 BEATS XGBoost on average (streaming).")
    else:
        print("=> XGBoost still ahead — but ŚŪNYATĀ improved adaptation. Tweak decay/blend to push further.")
    return df

# ---------------------------
# Main
# ---------------------------
def main():
    chunks, all_classes = load_data(n_chunks=8, chunk_size=5000)
    df = scenario_sunyata_v2(chunks, all_classes)
    os.makedirs('benchmark_results', exist_ok=True)
    df.to_csv('benchmark_results/sunyata_v2_streaming.csv', index=False)
    print("\nSaved results to benchmark_results/sunyata_v2_streaming.csv")

if __name__ == "__main__":
    main()
