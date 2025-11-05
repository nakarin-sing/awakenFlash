#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ŚŪNYATĀ v3 — Auto-adaptive NonLogic + RFF + Regularized Least Squares (streaming)
Goal: adapt decay/blend/sigma to try to beat streaming-like XGBoost.
Usage: drop into repo, run as python awakenFlash_benchmark_sunyata_v3.py
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# NonLogic meta-state (adaptive)
# -----------------------------
class NonLogicState:
    def __init__(self, n_features, blend=0.25, decay=0.92, amplify=1.5):
        self.blend = blend        # how strongly warm-start influences RLS initially
        self.decay = decay        # how fast the meta-state "forgets"
        self.amplify = amplify
        self.state = np.zeros(n_features, dtype=np.float32)
        self.initialized = False

    def fit(self, X, y):
        # simple signature: mean feature vector transformed
        sig = np.tanh(X.mean(axis=0) * self.amplify).astype(np.float32)
        if not self.initialized:
            self.state = sig.copy()
            self.initialized = True
        else:
            # blend current signature with stored state
            self.state = (1.0 - (1 - self.decay)) * self.state + (1.0 - self.decay) * sig
        # keep state normalized
        norm = np.linalg.norm(self.state) + 1e-12
        self.state = self.state / norm
        return self

    def get_warm(self, D):
        # expand/resize warm vector to dimension D (RFF dim)
        if not self.initialized:
            return None
        v = np.tile(self.state[:D // 2], 2) if len(self.state) >= D // 2 else np.resize(self.state, D)
        v = v.astype(np.float32)
        return v


# -----------------------------
# RFF + Dual RLS predictor (streaming)
# -----------------------------
class SunyataV3:
    def __init__(self, n_input_features, D=1024, ridge=10.0, gamma=0.08, seed=42):
        self.D = D
        self.ridge = float(ridge)
        self.gamma = float(gamma)   # affects RFF sampling variance
        self.rng = np.random.default_rng(seed)

        # RFF projection matrix (lazy init)
        self.W = self.rng.normal(0, np.sqrt(self.gamma), (self.D // 2, n_input_features)).astype(np.float32)
        self.bias = self.rng.uniform(0, 2 * np.pi, size=(self.D // 2,)).astype(np.float32)

        # model weights (alpha) — shape (D, K) after seen classes
        self.alpha = None
        self.classes_ = None
        self.class_to_idx = {}
        self.nl_state = NonLogicState(n_input_features, blend=0.30, decay=0.92, amplify=1.5)

        # adaptation metadata
        self.prev_acc = None
        self.window_acc = []
        self.adapt_counter = 0

    def _rff(self, X):
        proj = X @ self.W.T + self.bias  # (n, D/2)
        phi = np.concatenate([np.cos(proj), np.sin(proj)], axis=1) * np.sqrt(2.0 / self.D)
        return phi.astype(np.float32)

    def _ensure_label_map(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}

    def partial_fit(self, X, y, classes=None):
        # classes optional (from driver)
        if classes is not None:
            self.classes_ = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

        self._ensure_label_map(y)
        K = len(self.classes_)
        # update NonLogic meta-state
        self.nl_state.fit(X, y)
        warm = self.nl_state.get_warm(self.D)

        # features
        Phi = self._rff(X)   # (n, D)
        n, D = Phi.shape
        # encode labels
        y_idx = np.array([self.class_to_idx[int(v)] for v in y], dtype=np.int32)
        Y_onehot = np.zeros((n, K), dtype=np.float32)
        Y_onehot[np.arange(n), y_idx] = 1.0

        # compute normal equations pieces (small D vs huge N)
        PhiT_Phi = Phi.T @ Phi   # (D, D)
        PhiT_Y = Phi.T @ Y_onehot  # (D, K)

        H = PhiT_Phi + self.ridge * np.eye(D, dtype=np.float32)

        # warm-start expand
        if self.alpha is None:
            if warm is not None:
                # warm vector -> (D, K) small regular perturbation toward warm
                warm_exp = np.tile(warm[:, None], (1, K)).astype(np.float32)
                # regularized solve with warm bias
                self.alpha = np.linalg.solve(H, PhiT_Y + 1e-3 * warm_exp)
            else:
                self.alpha = np.linalg.solve(H, PhiT_Y)
        else:
            # incremental update strategy: solve new batch exactly but blend with old alpha
            try:
                new_alpha = np.linalg.solve(H, PhiT_Y)
            except np.linalg.LinAlgError:
                new_alpha = np.linalg.pinv(H) @ PhiT_Y
            # adapt blend factor based on meta-state blend
            blend = self.nl_state.blend
            self.alpha = (1 - blend) * self.alpha + blend * new_alpha

        return self

    def predict(self, X):
        if self.alpha is None:
            return np.zeros(len(X), dtype=int)
        Phi = self._rff(X)
        scores = Phi @ self.alpha  # (n, K)
        idxs = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in idxs], dtype=int)

    def adapt_on_feedback(self, acc):
        # keep a short moving window of acc
        self.window_acc.append(acc)
        if len(self.window_acc) > 5:
            self.window_acc.pop(0)

        # adjust meta-params when accuracy drops or stagnates
        if self.prev_acc is None:
            self.prev_acc = acc
            return

        # if performance dropped >7% compared to previous -> be more aggressive
        if acc < self.prev_acc * 0.93:
            # increase blend (use new batch more), reduce decay (remember more recent)
            self.nl_state.blend = min(0.6, self.nl_state.blend + 0.07)
            self.nl_state.decay = max(0.80, self.nl_state.decay - 0.03)
            self.gamma = max(0.01, self.gamma * 1.05)  # widen RFF variability a bit
            # rebuild W with updated gamma (cheap-ish)
            self.W = self.rng.normal(0, np.sqrt(self.gamma), self.W.shape).astype(np.float32)
            self.adapt_counter += 1
        else:
            # if improving slowly, gently reduce blend so model stabilizes
            if acc > self.prev_acc:
                self.nl_state.blend = max(0.05, self.nl_state.blend * 0.95)
                self.nl_state.decay = min(0.98, self.nl_state.decay * 1.01)
                self.gamma = max(0.005, self.gamma * 0.995)

        # occasionally nudge ridge to avoid over/under regularization
        if np.std(self.window_acc) > 0.03:
            self.ridge = max(0.5, self.ridge * 0.95)
        else:
            self.ridge = min(200.0, self.ridge * 1.01)

        self.prev_acc = acc


# -----------------------------
# data loader (small quick chunks by default)
# -----------------------------
def load_data(n_chunks=8, chunk_size=4000, nrows=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("Loading dataset (may take a moment)...")
    df = pd.read_csv(url, header=None, nrows=nrows or n_chunks * chunk_size)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int32)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)
    chunks = [(X_all[i:i + chunk_size], y_all[i:i + chunk_size]) for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)


# -----------------------------
# benchmark driver: streaming compare
# -----------------------------
def streaming_benchmark(chunks, all_classes):
    n_input = chunks[0][0].shape[1]
    model = SunyataV3(n_input, D=1024, ridge=20.0, gamma=0.08, seed=42)

    results = []
    for i, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        print(f"Chunk {i:02d}/{len(chunks)} | train={len(X_train)} test={len(X_test)}")
        # train & predict
        t0 = time.time()
        model.partial_fit(X_train, y_train, classes=all_classes)
        t_fit = time.time() - t0

        t0 = time.time()
        pred = model.predict(X_test)
        t_pred = time.time() - t0
        acc = accuracy_score(y_test, pred)

        # xgboost streaming-ish baseline (small window training)
        t0 = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        xgbm = xgb.train({"objective": "multi:softmax", "num_class": len(all_classes),
                          "max_depth": 4, "eta": 0.2, "verbosity": 0}, dtrain, num_boost_round=10)
        xgb_pred = xgbm.predict(dtest).astype(int)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        t_xgb = time.time() - t0

        print(f"  ŚŪNYATĀ: acc={acc:.3f} fit={t_fit:.3f} pred={t_pred:.3f}")
        print(f"  XGB:     acc={xgb_acc:.3f} time={t_xgb:.3f}")

        results.append({
            "chunk": i,
            "sunyata_acc": acc,
            "sunyata_fit": t_fit,
            "sunyata_pred": t_pred,
            "xgb_acc": xgb_acc,
            "xgb_time": t_xgb,
            "blend": model.nl_state.blend,
            "decay": model.nl_state.decay,
            "gamma": model.gamma,
            "ridge": model.ridge
        })

        # adapt model meta state using observed accuracy
        model.adapt_on_feedback(acc)
        print(f"    -> adapted blend={model.nl_state.blend:.3f} decay={model.nl_state.decay:.3f} ridge={model.ridge:.3f}\n")

    df = pd.DataFrame(results)
    print("\nFINAL SUMMARY")
    print("ŚŪNYATĀ avg acc:", df["sunyata_acc"].mean(), "| avg fit:", df["sunyata_fit"].mean())
    print("XGB     avg acc:", df["xgb_acc"].mean(), "| avg time:", df["xgb_time"].mean())
    return df


# -----------------------------
# main
# -----------------------------
def main():
    chunks, classes = load_data(n_chunks=8, chunk_size=4000)
    res = streaming_benchmark(chunks, classes)
    os.makedirs("benchmark_results", exist_ok=True)
    res.to_csv("benchmark_results/sunyata_v3_streaming.csv", index=False)
    print("Saved results to benchmark_results/sunyata_v3_streaming.csv")


if __name__ == "__main__":
    main()
