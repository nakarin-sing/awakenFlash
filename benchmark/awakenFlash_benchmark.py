#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark_v2_nonlogic.py
awakenFlash + Non-Logic Optimizer Core (streaming adaptive blend)
- streaming chunks
- Sunyata (RFF + ridge) as fast online learner
- XGBoost as teacher / strong baseline
- NonLogicOptimizer: adapt blend weights, detect drift, temperature scaling
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")


# ---------------------------
# Lightweight Sunyata model (RFF + ridge)
# ---------------------------
class FastSunyata:
    def __init__(self, D=512, ridge=1.0, gamma=0.1, rng_seed=42):
        self.D = int(D)
        self.ridge = float(ridge)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(rng_seed)
        self.W = None
        self.alpha = None
        self.classes_ = None

    def _ensure_W(self, n_features):
        if self.W is None:
            rows = self.D // 2
            self.W = self.rng.normal(0, np.sqrt(self.gamma), (rows, n_features)).astype(np.float32)

    def _rff(self, X):
        self._ensure_W(X.shape[1])
        proj = X @ self.W.T  # (n, D/2)
        phi = np.hstack([np.cos(proj), np.sin(proj)]) * np.sqrt(2.0 / self.D)
        return phi.astype(np.float32)

    def partial_fit(self, X, y, classes=None):
        Xf = X.astype(np.float32)
        phi = self._rff(Xf)
        if classes is not None:
            self.classes_ = np.array(classes)
        if self.classes_ is None:
            self.classes_ = np.unique(y)
        K = len(self.classes_)
        n = phi.shape[0]
        y_idx = np.searchsorted(self.classes_, y)
        Y = np.zeros((n, K), dtype=np.float32)
        Y[np.arange(n), y_idx] = 1.0

        # closed-form ridge for multi-output
        PhiTPhi = (phi.T @ phi) / max(1, n)
        PhiTy = (phi.T @ Y) / max(1, n)
        H = PhiTPhi + np.eye(PhiTPhi.shape[0], dtype=np.float32) * (self.ridge / max(1, n))
        try:
            self.alpha = np.linalg.solve(H + 1e-6*np.eye(H.shape[0]), PhiTy)
        except np.linalg.LinAlgError:
            self.alpha = np.linalg.pinv(H + 1e-6*np.eye(H.shape[0])) @ PhiTy

    def predict_proba(self, X):
        if self.alpha is None:
            # uniform fallback
            n = X.shape[0]
            K = len(self.classes_) if self.classes_ is not None else 1
            return np.ones((n, K), dtype=np.float32) / max(1, K)
        phi = self._rff(X.astype(np.float32))
        scores = phi @ self.alpha  # (n, K)
        # softmax stable
        logits = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(np.clip(logits, -50, 50))
        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]


# ---------------------------
# Non-Logic Optimizer Core
# - maintains blend weight between Sunyata and XGB
# - detects drift by comparing short-term acc vs long-term acc
# - adjusts temperature and teacher weight
# ---------------------------
class NonLogicOptimizer:
    def __init__(self, init_blend=0.5, lr=0.12, drift_window=3):
        self.blend = float(init_blend)  # weight for Sunyata (0..1), final prediction = blend * sun + (1-blend) * xgb
        self.lr = float(lr)
        self.drift_window = int(drift_window)
        self.history = []  # keep last few (s_acc, x_acc)
        self.ema_s_acc = None
        self.ema_x_acc = None
        self.ema_alpha = 0.2

        # temperature for soft labels mixing
        self.temp = 2.0
        self.teacher_soft_weight = 0.25

    def update(self, sun_acc, xgb_acc):
        # update EMA
        if self.ema_s_acc is None:
            self.ema_s_acc = sun_acc
            self.ema_x_acc = xgb_acc
        else:
            self.ema_s_acc = self.ema_alpha * sun_acc + (1 - self.ema_alpha) * self.ema_s_acc
            self.ema_x_acc = self.ema_alpha * xgb_acc + (1 - self.ema_alpha) * self.ema_x_acc

        # push history
        self.history.append((sun_acc, xgb_acc))
        if len(self.history) > 50:
            self.history.pop(0)

        # simple drift detection: if recent drop in xgb but sun stays stable -> increase blend
        last_k = self.history[-self.drift_window:]
        if len(last_k) >= self.drift_window:
            s_mean = np.mean([s for s, _ in last_k])
            x_mean = np.mean([x for _, x in last_k])
            # if xgb degraded relatively to sunyata -> move blend towards sunyata
            diff = (s_mean - x_mean)
            # adjust more conservatively when diff small
            adapt = np.tanh(diff * 5.0) * self.lr
            self.blend = np.clip(self.blend + adapt, 0.0, 1.0)

        # baseline: if sun_acc > xgb_acc for long-term EMA -> favor sunyata
        delta = self.ema_s_acc - self.ema_x_acc
        self.blend = np.clip(self.blend + np.sign(delta) * self.lr * (abs(delta) ** 0.7), 0.0, 1.0)

        # adapt temperature and teacher weight modestly
        if self.blend > 0.7:
            self.temp = max(1.0, self.temp * 0.96)
            self.teacher_soft_weight = max(0.05, self.teacher_soft_weight * 0.95)
        elif self.blend < 0.3:
            self.temp = min(5.0, self.temp * 1.03)
            self.teacher_soft_weight = min(0.6, self.teacher_soft_weight * 1.05)

    def get_blend(self):
        return float(self.blend)

    def get_teacher_weight(self):
        return float(self.teacher_soft_weight)

    def get_temp(self):
        return float(self.temp)


# ---------------------------
# Utilities: soft blending of probability vectors
# ---------------------------
def blend_probs(p_sun, p_xgb, blend):
    # p_sun, p_xgb shape (n, K)
    w = np.clip(blend, 0.0, 1.0)
    return w * p_sun + (1 - w) * p_xgb


# ---------------------------
# Load data (covtype streaming chunks)
# ---------------------------
def load_data(n_chunks=10, chunk_size=4000, random_state=42):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("Loading data (this may take a while)...")
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = (df.iloc[:, -1].values - 1).astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    chunks = [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, len(X), chunk_size)]
    classes = np.unique(y)
    return chunks[:n_chunks], classes


# ---------------------------
# Streaming benchmark using optimizer
# ---------------------------
def scenario_streaming_nonlogic(chunks, classes):
    sunyata = FastSunyata(D=512, ridge=10.0, gamma=0.05)
    optimizer = NonLogicOptimizer(init_blend=0.5, lr=0.12, drift_window=3)

    results = []
    xgb_window_X = []
    xgb_window_y = []
    X_window_size = 3  # number of chunks kept for xgb train to simulate streaming teacher

    for i, (X_chunk, y_chunk) in enumerate(chunks, 1):
        n = X_chunk.shape[0]
        split = int(0.8 * n)
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # update sunyata quickly
        sunyata.partial_fit(X_train, y_train, classes=classes)
        p_sun = sunyata.predict_proba(X_test)
        pred_s = np.argmax(p_sun, axis=1)
        s_acc = accuracy_score(y_test, pred_s)

        # build/maintain a small xgb teacher on recent chunks
        xgb_window_X.append(X_train)
        xgb_window_y.append(y_train)
        if len(xgb_window_X) > X_window_size:
            xgb_window_X = xgb_window_X[-X_window_size:]
            xgb_window_y = xgb_window_y[-X_window_size:]

        X_xgb_train = np.vstack(xgb_window_X)
        y_xgb_train = np.concatenate(xgb_window_y)

        # train lightweight xgb on window
        dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train(
            {"objective": "multi:softprob", "num_class": len(classes),
             "max_depth": 3, "eta": 0.25, "verbosity": 0}, dtrain, num_boost_round=6
        )
        p_xgb = xgb_model.predict(dtest)  # softprob
        # ensure shape: (n, K)
        p_xgb = np.asarray(p_xgb)
        if p_xgb.ndim == 1:
            # sometimes returns class indices; fallback to one-hot
            onehot = np.zeros((len(p_xgb), len(classes)), dtype=np.float32)
            onehot[np.arange(len(p_xgb)), p_xgb.astype(int)] = 1.0
            p_xgb = onehot
        pred_x = np.argmax(p_xgb, axis=1)
        x_acc = accuracy_score(y_test, pred_x)

        # optimizer updates blend
        optimizer.update(s_acc, x_acc)
        blend = optimizer.get_blend()

        # optionally incorporate teacher soft labels into sunyata target (mimic distillation)
        # produce blended output
        p_blend = blend_probs(p_sun, p_xgb, blend)
        pred_blend = np.argmax(p_blend, axis=1)
        blend_acc = accuracy_score(y_test, pred_blend)
        blend_f1 = f1_score(y_test, pred_blend, average="weighted")

        results.append({
            "chunk": i,
            "sun_acc": s_acc,
            "xgb_acc": x_acc,
            "blend_acc": blend_acc,
            "blend_f1": blend_f1,
            "blend_weight": blend,
            "ema_s": optimizer.ema_s_acc,
            "ema_x": optimizer.ema_x_acc
        })

        print(f"Chunk {i:02d} | SUN={s_acc:.3f} XGB={x_acc:.3f} BLEND={blend_acc:.3f} w={blend:.3f}")

    df = pd.DataFrame(results)
    print("\nSTREAMING SUMMARY")
    print(df[["sun_acc", "xgb_acc", "blend_acc"]].mean())
    return df


# ---------------------------
# Main
# ---------------------------
def main():
    print("awakenFlash v2 nonlogic benchmark start")
    chunks, classes = load_data(n_chunks=10, chunk_size=4000)
    df = scenario_streaming_nonlogic(chunks, classes)
    os.makedirs("benchmark_results", exist_ok=True)
    df.to_csv("benchmark_results/awakenFlash_v2_nonlogic_streaming.csv", index=False)
    print("Saved results to benchmark_results/awakenFlash_v2_nonlogic_streaming.csv")


if __name__ == "__main__":
    main()
