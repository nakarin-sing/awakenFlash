#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_anlk_benchmark.py
Prototype: Adaptive Non-Logic Kernel (ANLK) streaming benchmark vs XGBoost
Author: generated
"""

import os
import time
import math
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Utility functions
# ---------------------------
def load_covtype(n_chunks=10, chunk_size=5000, random_state=42):
    """Load UCI covtype (downloaded live). Keep it simple and chunked."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("Loading dataset (this may take a bit)...")
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = (df.iloc[:, -1].values - 1).astype(np.int32)  # 0-based classes
    X, y = shuffle(X, y, random_state=random_state)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    chunks = [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, len(X), chunk_size)]
    return chunks[:n_chunks], np.unique(y)

def entropy_of_probs(probs):
    # probs: (n_samples, n_classes)
    p = np.clip(probs, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=1)
    return ent.mean()

def model_memory_snapshot():
    current, peak = tracemalloc.get_traced_memory()
    return current/1024/1024, peak/1024/1024

# ---------------------------
# AwakenFlash: ANLK streaming model
# ---------------------------
class AwakenFlashANLK:
    """
    Adaptive Non-Logic Kernel (ANLK) prototype.
    - lightweight RFF + ridge solving per chunk (fast)
    - meta-weight w_t adapts using delta_acc, entropy, drift
    - returns class predictions
    """
    def __init__(self, D=512, ridge=10.0, forgetting=0.98, seed=0):
        self.D = int(D)
        self.ridge = float(ridge)
        self.forgetting = float(forgetting)
        self.rng = np.random.default_rng(seed)
        self.W = None  # RFF projection
        self.alpha = None  # (D, K)
        self.classes_ = None
        self.class_to_idx = {}
        self.prev_acc = None
        self.prev_chunk_mean = None
        self.w = 0.5  # blend weight (0..1): 1 -> fully nonlogic, 0 -> fallback linear
        # meta params
        self.alpha_meta = 3.0
        self.beta_meta = 1.0
        self.gamma_meta = 2.0

    def _ensure_W(self, n_features):
        half = self.D // 2
        if self.W is None or self.W.shape[1] != n_features:
            scale = 1.0 / math.sqrt(max(1, n_features))
            self.W = self.rng.normal(0, scale, (half, n_features)).astype(np.float32)

    def _rff(self, X):
        self._ensure_W(X.shape[1])
        proj = X.astype(np.float32) @ self.W.T
        phi = np.hstack([np.cos(proj), np.sin(proj)]) * math.sqrt(2.0 / self.D)
        return phi  # (n, D)

    def _encode(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            self.class_to_idx = {c:i for i,c in enumerate(self.classes_)}
        return np.array([self.class_to_idx[v] for v in y], dtype=np.int32)

    def _update_meta_weight(self, delta_acc, entropy, drift):
        # logistic/sigmoid rule: w ← sigmoid(α*Δacc + β*entropy - γ*drift)
        val = self.alpha_meta * (delta_acc if delta_acc is not None else 0.0) + \
              self.beta_meta * (entropy) - \
              self.gamma_meta * (drift)
        w_new = 1.0 / (1.0 + math.exp(-val))
        # smooth update
        self.w = 0.85 * self.w + 0.15 * w_new
        # clamp
        self.w = float(min(0.99, max(0.01, self.w)))

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
            self.class_to_idx = {c:i for i,c in enumerate(classes)}
        K = len(self.classes_)

        phi = self._rff(X)  # (n,D)
        y_idx = self._encode(y)
        n, D = phi.shape

        # compute simple teacher probs from current linear model (if exists) for entropy
        if self.alpha is not None:
            scores = phi @ self.alpha  # (n, K)
            # softmax
            exps = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs = exps / exps.sum(axis=1, keepdims=True)
            ent = entropy_of_probs(probs)
        else:
            ent = math.log(max(2, K))  # maximal initial entropy proxy

        # compute drift: distance between current chunk mean and prev chunk mean (feature-wise)
        cur_mean = X.mean(axis=0)
        if self.prev_chunk_mean is None:
            drift = 0.0
        else:
            drift = float(np.linalg.norm(cur_mean - self.prev_chunk_mean) / (np.linalg.norm(self.prev_chunk_mean)+1e-9))
        self.prev_chunk_mean = cur_mean.copy()

        # delta_acc: if we have prev_acc
        delta_acc = None
        # We'll compute delta_acc after predict() externally; here update meta with None or 0

        # Prepare one-hot
        y_onehot = np.zeros((n, K), dtype=np.float32)
        y_onehot[np.arange(n), y_idx] = 1.0

        # RLS-like solve with forgetting
        PhiT_Phi = (phi.T @ phi) / max(1, n)
        PhiT_y = (phi.T @ y_onehot) / max(1, n)
        ridge = self.ridge / max(1, n)
        H = PhiT_Phi + np.eye(D, dtype=np.float32) * ridge

        if self.alpha is None:
            # initialization
            self.alpha = np.linalg.solve(H + 1e-6*np.eye(D), PhiT_y)
        else:
            # blend with forgetting to simulate continual update
            rhs = PhiT_y + self.forgetting * self.alpha
            try:
                self.alpha = np.linalg.solve(H + 1e-6*np.eye(D), rhs)
            except np.linalg.LinAlgError:
                self.alpha = np.linalg.pinv(H + 1e-6*np.eye(D)) @ rhs

        # return entropy and drift for meta update
        return ent, drift

    def predict(self, X):
        if self.alpha is None:
            # bootstrap: random predicts (rare)
            return np.zeros(len(X), dtype=self.classes_.dtype)
        phi = self._rff(X)
        scores = phi @ self.alpha  # (n,K)
        pred_idx = np.argmax(scores, axis=1)
        return self.classes_[pred_idx]

# ---------------------------
# Benchmark runner
# ---------------------------
def compute_adaptivity_stream(score_series, window=3):
    # rough adaptivity proxy: how quickly accuracy recovers after drops
    # find local drops and measure recovery within window chunks
    s = np.asarray(score_series)
    drops = []
    for i in range(1, len(s)):
        if s[i] < s[i-1] - 0.02:  # dropped more than 2%
            # find next index within window where s >= previous level
            recovered = False
            for j in range(i+1, min(len(s), i+1+window)):
                if s[j] >= s[i-1]:
                    drops.append((i, j, (j - i)))
                    recovered = True
                    break
            if not recovered:
                drops.append((i, None, None))
    if not drops:
        return 1.0  # no drops -> very adaptive (or stable)
    # adaptivity score: inverse of avg recovery time (normalized)
    times = [d[2] for d in drops if d[2] is not None]
    if not times:
        return 0.2
    avg = np.mean(times)
    return max(0.0, 1.0 - avg/(window+1))

def run_benchmark(n_chunks=10, chunk_size=5000, random_state=42, out_dir='benchmark_results'):
    os.makedirs(out_dir, exist_ok=True)
    tracemalloc.start()

    # load chunks
    chunks, classes = load_covtype(n_chunks=n_chunks, chunk_size=chunk_size, random_state=random_state)

    # models
    sun = AwakenFlashANLK(D=512, ridge=20.0, forgetting=0.97, seed=random_state)
    xgb_params = {"objective":"multi:softmax", "num_class": len(classes), "max_depth":4, "eta":0.2, "verbosity":0}
    results = []

    # For blend baseline (simple blend of SUN and XGB)
    blend_weights = []

    sun_prev_acc = None
    for cid, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # 1) AwakenFlash partial_fit (streaming)
        t0 = time.time()
        ent, drift = sun.partial_fit(X_train, y_train, classes=classes)
        pred_s = sun.predict(X_test)
        t_s = time.time() - t0
        acc_s = accuracy_score(y_test, pred_s)
        f1_s = f1_score(y_test, pred_s, average='weighted')

        # compute delta_acc and update meta weight
        delta_acc = None
        if sun_prev_acc is None:
            delta_acc = 0.0
        else:
            delta_acc = acc_s - sun_prev_acc
        sun._update_meta_weight(delta_acc, ent, drift)
        sun_prev_acc = acc_s

        # 2) XGBoost train-per-chunk (streaming-like baseline)
        t0 = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=8)
        pred_x = xgb_model.predict(dtest).astype(int)
        t_x = time.time() - t0
        acc_x = accuracy_score(y_test, pred_x)
        f1_x = f1_score(y_test, pred_x, average='weighted')

        # 3) Blend baseline: weighted voting using sun.w (proxy)
        # we'll use sun.w to blend predictions (sun stronger -> weight toward sun)
        # get soft proxies: for sun, use one-hot of pred; for xgb, same
        # (cheap approximate)
        w = sun.w
        blend_pred = np.where(np.random.rand(len(pred_s)) < w, pred_s, pred_x)
        acc_b = accuracy_score(y_test, blend_pred)

        # memory snapshot
        mem_cur, mem_peak = model_memory_snapshot()

        print(f"Chunk {cid:02d} | SUN={acc_s:.3f} XGB={acc_x:.3f} BLEND={acc_b:.3f} w={w:.3f} ent={ent:.3f} drift={drift:.4f}")
        results.append({
            'chunk': cid,
            'sun_acc': acc_s, 'sun_f1': f1_s, 'sun_time': t_s,
            'xgb_acc': acc_x, 'xgb_f1': f1_x, 'xgb_time': t_x,
            'blend_acc': acc_b, 'blend_w': w,
            'entropy': ent, 'drift': drift,
            'mem_mb': mem_cur, 'mem_peak_mb': mem_peak
        })
        blend_weights.append(w)

    df = pd.DataFrame(results)
    # summary metrics
    summary = {}
    summary['sun_mean_acc'] = df['sun_acc'].mean()
    summary['xgb_mean_acc'] = df['xgb_acc'].mean()
    summary['blend_mean_acc'] = df['blend_acc'].mean()
    summary['sun_mean_f1'] = df['sun_f1'].mean()
    summary['xgb_mean_f1'] = df['xgb_f1'].mean()
    summary['sun_mean_time'] = df['sun_time'].mean()
    summary['xgb_mean_time'] = df['xgb_time'].mean()
    summary['sun_mem_mb_mean'] = df['mem_mb'].mean()
    summary['sun_mem_peak_mb'] = df['mem_peak_mb'].max()
    # adaptivity
    summary['sun_adaptivity'] = compute_adaptivity_stream(df['sun_acc'].values)
    summary['xgb_adaptivity'] = compute_adaptivity_stream(df['xgb_acc'].values)
    # stability: lower var means more stable
    summary['sun_stability_var'] = float(np.var(df['sun_acc'].values))
    summary['xgb_stability_var'] = float(np.var(df['xgb_acc'].values))
    # interpretability proxy: entropy of blend weights
    hist, _ = np.histogram(blend_weights, bins=10, range=(0,1), density=True)
    hist = np.clip(hist, 1e-12, 1.0)
    summary['interpretability_flow_entropy'] = float(-np.sum(hist * np.log(hist)))

    # write outputs
    df.to_csv(os.path.join(out_dir, 'streaming_per_chunk_results.csv'), index=False)
    pd.Series(summary).to_csv(os.path.join(out_dir, 'summary_metrics.csv'))

    # plots
    plt.figure(figsize=(10,4))
    plt.plot(df['chunk'], df['sun_acc'], 'o-', label='AwakenFlash (SUN)')
    plt.plot(df['chunk'], df['xgb_acc'], 's-', label='XGBoost (per-chunk)')
    plt.plot(df['chunk'], df['blend_acc'], '^-', label='Blend')
    plt.xlabel('Chunk')
    plt.ylabel('Accuracy')
    plt.title('Streaming accuracy per chunk')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'streaming_accuracy.png'), dpi=150)
    plt.close()

    print("\nSUMMARY")
    for k,v in summary.items():
        print(f"  {k}: {v}")

    print(f"\nSaved results in {out_dir}")
    tracemalloc.stop()
    return df, summary

# ---------------------------
# CLI run
# ---------------------------
if __name__ == "__main__":
    df, summary = run_benchmark(n_chunks=10, chunk_size=4000, random_state=123, out_dir='benchmark_results')
