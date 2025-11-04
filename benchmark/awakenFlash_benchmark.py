#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_scenario4_adaptive.py
Scenario 4 — Online Adaptation & Drift Recovery Test
- Stream chunks from Covtype (or local CSV if you prefer)
- Simulate a distribution shift (drift) at configurable chunk index
- Run online models and an adaptive protocol:
    * baseline partial_fit per chunk
    * detect drift when recent accuracy drops below threshold
    * on drift: replay recent buffer and/or increase update intensity
- Save CSV of per-chunk metrics and a PNG plot of accuracy timeline.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb

# ------------- config -------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
N_CHUNKS = 12
CHUNK_SIZE = 10000
DRIFT_AT_CHUNK = 7          # simulate drift starting at this chunk (1-indexed)
DRIFT_MAGNITUDE = 1.5       # multiplier for noise added to features after drift (larger -> stronger drift)
DRIFT_LABEL_SHIFT = 0       # if >0, shift labels (not used here)
REPLAY_WINDOW = 3           # how many recent chunks to replay when drift detected
DRIFT_DETECT_WINDOW = 3     # rolling window size for drift detector (accuracy)
DRIFT_DROP_THRESHOLD = 0.10 # if accuracy drops by > this (absolute), trigger adaptation
ADAPTIVE_PASSES = 5         # number of extra partial_fit passes on replay data during adaptation
RESULTS_DIR = "benchmark_results_s4"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------- helpers -------------
def load_and_prepare(limit_chunks=N_CHUNKS, chunk_size=CHUNK_SIZE, drift_at=DRIFT_AT_CHUNK):
    print("Loading dataset (this may take a few seconds)...")
    df = pd.read_csv(DATA_URL, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = (df.iloc[:, -1].values - 1).astype(int)  # convert to 0..6
    # scale globally (for fairness)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # split into chunks
    chunks = []
    n_rows = X.shape[0]
    max_chunks = min(limit_chunks, (n_rows // chunk_size))
    for i in range(max_chunks):
        start = i * chunk_size
        end = start + chunk_size
        Xi = X[start:end].copy()
        yi = y[start:end].copy()
        # simulate drift by adding scaled gaussian noise AFTER drift point
        if (i + 1) >= drift_at:
            Xi += np.random.normal(loc=0.0, scale=DRIFT_MAGNITUDE, size=Xi.shape)
        chunks.append((Xi, yi))
    classes = np.unique(y)
    return chunks, classes

# simple rolling average utility
def rolling_mean(a, window):
    if len(a) < window:
        return np.mean(a) if len(a) else 0.0
    return np.mean(a[-window:])

# ------------- Online RLS (simple stable variant) -------------
class SimpleOnlineRLS:
    """A lightweight RLS-like online linear classifier for multiclass one-vs-rest.
       Not highly optimized but small and stable for streaming tests.
    """
    def __init__(self, n_features, n_classes, reg=1e-2, forget=0.999):
        self.nf = n_features
        self.nc = n_classes
        self.forget = forget
        self.reg = reg
        # weight matrix shape (nf+1, nc) with bias included
        self.W = np.zeros((self.nf + 1, self.nc), dtype=np.float32)
        # P is (nf+1, nf+1) covariance approx
        self.P = np.eye(self.nf + 1, dtype=np.float32) / self.reg

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X.astype(np.float32)])

    def partial_fit(self, X, y):
        Xb = self._add_bias(X)
        # one-vs-rest update
        for i in range(Xb.shape[0]):
            xi = Xb[i:i+1].T  # (d,1)
            yi = np.zeros((1, self.nc), dtype=np.float32)
            yi[0, y[i]] = 1.0
            # predict current
            pred = (xi.T @ self.W).ravel()
            # vector error
            e = (yi.ravel() - pred).reshape(-1, 1)  # (nc,1)
            # update P with forgetting via Sherman-Morrison-ish step (stable small-batch)
            denom = (self.forget + (xi.T @ self.P @ xi)).item()
            k = (self.P @ xi) / denom  # (d,1)
            # update W for all classes simultaneously
            self.W += (k @ e.T)
            # update P
            self.P = (self.P - (k @ (xi.T @ self.P))) / self.forget

    def predict(self, X):
        Xb = self._add_bias(X)
        logits = Xb @ self.W
        return logits.argmax(axis=1)

# ------------- main scenario 4 test -------------
def scenario4_adaptive():
    chunks, classes = load_and_prepare()
    n_classes = len(classes)
    n_features = chunks[0][0].shape[1]

    # models
    sgd = SGDClassifier(loss="log_loss", learning_rate="adaptive", max_iter=1, warm_start=True, random_state=42)
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=1, warm_start=True, random_state=42)
    rls = SimpleOnlineRLS(n_features=n_features, n_classes=n_classes, reg=1e-2, forget=0.995)

    first_sgd = first_pa = True

    # replay buffer for adaptation (store last k train chunks)
    replay_buffer_X = []
    replay_buffer_y = []

    # history logging
    history = {
        'chunk': [], 'sgd_acc': [], 'pa_acc': [], 'rls_acc': [],
        'sgd_time': [], 'pa_time': [], 'rls_time': [],
        'drift_flag': []
    }

    sgd_acc_history = []
    pa_acc_history = []
    rls_acc_history = []

    for idx, (X_chunk, y_chunk) in enumerate(chunks, start=1):
        # split chunk
        split = int(0.8 * len(X_chunk))
        X_tr, X_te = X_chunk[:split], X_chunk[split:]
        y_tr, y_te = y_chunk[:split], y_chunk[split:]

        # append to replay buffer
        replay_buffer_X.append((X_tr.copy(), y_tr.copy()))
        if len(replay_buffer_X) > REPLAY_WINDOW:
            replay_buffer_X.pop(0)

        # ---- baseline online updates ----
        # SGD
        t0 = time.time()
        if first_sgd:
            sgd.partial_fit(X_tr, y_tr, classes=classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_tr, y_tr)
        t_sgd = time.time() - t0
        sgd_pred = sgd.predict(X_te)
        sgd_acc = accuracy_score(y_te, sgd_pred)

        # PA
        t0 = time.time()
        if first_pa:
            pa.partial_fit(X_tr, y_tr, classes=classes)
            first_pa = False
        else:
            pa.partial_fit(X_tr, y_tr)
        t_pa = time.time() - t0
        pa_pred = pa.predict(X_te)
        pa_acc = accuracy_score(y_te, pa_pred)

        # RLS
        t0 = time.time()
        rls.partial_fit(X_tr, y_tr)
        t_rls = time.time() - t0
        rls_pred = rls.predict(X_te)
        rls_acc = accuracy_score(y_te, rls_pred)

        # log
        history['chunk'].append(idx)
        history['sgd_acc'].append(sgd_acc)
        history['pa_acc'].append(pa_acc)
        history['rls_acc'].append(rls_acc)
        history['sgd_time'].append(t_sgd)
        history['pa_time'].append(t_pa)
        history['rls_time'].append(t_rls)
        history['drift_flag'].append(False)

        sgd_acc_history.append(sgd_acc)
        pa_acc_history.append(pa_acc)
        rls_acc_history.append(rls_acc)

        # ---- drift detection (simple) ----
        # measure recent performance drop for each model vs its own recent avg
        drift_detected = False
        if idx >= DRIFT_DETECT_WINDOW + 1:
            sgd_prev = rolling_mean(sgd_acc_history[:-1], DRIFT_DETECT_WINDOW)
            pa_prev = rolling_mean(pa_acc_history[:-1], DRIFT_DETECT_WINDOW)
            rls_prev = rolling_mean(rls_acc_history[:-1], DRIFT_DETECT_WINDOW)
            # if any model drops sharply relative to its recent avg -> drift
            if (sgd_prev - sgd_acc) > DRIFT_DROP_THRESHOLD or \
               (pa_prev - pa_acc) > DRIFT_DROP_THRESHOLD or \
               (rls_prev - rls_acc) > DRIFT_DROP_THRESHOLD:
                drift_detected = True

        # ---- adaptation protocol ----
        if drift_detected:
            print(f"*** Drift detected at chunk {idx} — running adaptation (replay + aggressive updates)")
            history['drift_flag'][-1] = True
            # prepare replay data from buffer
            X_replay = np.vstack([xb for xb, yb in replay_buffer_X])
            y_replay = np.hstack([yb for xb, yb in replay_buffer_X])
            # aggressive partial_fit: multiple passes over replay
            for pass_i in range(ADAPTIVE_PASSES):
                sgd.partial_fit(X_replay, y_replay)
                pa.partial_fit(X_replay, y_replay)
                rls.partial_fit(X_replay, y_replay)

        print(f"Chunk {idx:02d}: SGD={sgd_acc:.3f} PA={pa_acc:.3f} RLS={rls_acc:.3f} "
              f"{'(DRIFT)' if drift_detected else ''}")

    # save results CSV
    df = pd.DataFrame(history)
    csv_path = os.path.join(RESULTS_DIR, "scenario4_adaptive_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    # plot timeline
    plt.figure(figsize=(10, 5))
    plt.plot(df['chunk'], df['sgd_acc'], marker='o', label='SGD (online)')
    plt.plot(df['chunk'], df['pa_acc'], marker='s', label='PA (online)')
    plt.plot(df['chunk'], df['rls_acc'], marker='^', label='RLS (online)')
    # mark drift flags
    drift_chunks = df[df['drift_flag'] == True]['chunk'].tolist()
    for dc in drift_chunks:
        plt.axvline(dc, color='red', linestyle='--', alpha=0.4)
    plt.xlabel('chunk')
    plt.ylabel('accuracy')
    plt.title('Scenario 4 — Online Adaptation & Drift Recovery')
    plt.legend()
    plt.grid(True)
    png_path = os.path.join(RESULTS_DIR, "scenario4_adaptive_acc_timeline.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot → {png_path}")

    print("\nDone.")

if __name__ == "__main__":
    scenario4_adaptive()
