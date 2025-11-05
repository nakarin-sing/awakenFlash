#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – SunyataV3 (Non-Logic warm-start + replay) vs streaming XGBoost
Use: python awakenFlash_benchmark.py
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


# -------------------------
# NonLogicState (warm-start/meta)
# -------------------------
class NonLogicState:
    def __init__(self, n_features, blend=0.30, decay=0.92, amplify=1.5):
        self.nf = n_features
        self.blend = float(blend)
        self.decay = float(decay)
        self.amplify = float(amplify)
        self.state = None

    def fit(self, X, y):
        # simple "signature" representing dataset moment
        sig = np.tanh(X.mean(axis=0) * self.amplify).astype(np.float32)
        if self.state is None:
            self.state = sig
        else:
            self.state = self.state * (1.0 - 0.2 * (1.0 - self.decay)) + sig * (0.2 * (1.0 - self.decay))
        # slightly adapt blend with stability
        if np.std(X) > 0.0:
            self.blend = max(0.01, min(0.9, self.blend * (1.0 - (0.001 * np.std(X)))))
        return self

    def get_warm(self, D):
        # expand or compress state to length D
        if self.state is None:
            return None
        s = self.state
        if len(s) >= D:
            return s[:D].astype(np.float32)
        else:
            # tile until length D, then trim
            reps = int(np.ceil(D / len(s)))
            out = np.tile(s, reps)[:D].astype(np.float32)
            return out


# -------------------------
# SunyataV3 (main model)
# -------------------------
class SunyataV3:
    def __init__(self, n_input_features, D=2048, ridge=5.0, gamma=0.08, seed=42,
                 hard_buffer_size=2000, replay_mix=0.30):
        self.D = int(D)
        self.ridge = float(ridge)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, np.sqrt(self.gamma), (self.D // 2, n_input_features)).astype(np.float32)
        self.bias = self.rng.uniform(0, 2 * np.pi, size=(self.D // 2,)).astype(np.float32)
        self.alpha = None
        self.classes_ = None
        self.class_to_idx = {}
        self.nl_state = NonLogicState(n_input_features, blend=0.30, decay=0.92, amplify=1.5)
        self.prev_acc = None
        self.window_acc = []
        self.adapt_counter = 0
        # Hard-example replay buffer
        self.hard_X = np.empty((0, n_input_features), dtype=np.float32)
        self.hard_y = np.empty((0,), dtype=np.int32)
        self.hard_buffer_size = int(hard_buffer_size)
        self.replay_mix = float(replay_mix)

    def _rff(self, X):
        proj = (X @ self.W.T) + self.bias  # shape (n, D/2)
        phi = np.concatenate([np.cos(proj), np.sin(proj)], axis=1) * np.sqrt(2.0 / self.D)
        return phi.astype(np.float32)

    def _ensure_label_map(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}

    def _add_hard_examples(self, X_mis, y_mis):
        if X_mis is None or len(X_mis) == 0:
            return
        X_mis = X_mis.astype(np.float32)
        y_mis = y_mis.astype(np.int32)
        if self.hard_X.size == 0:
            self.hard_X = X_mis.copy()
            self.hard_y = y_mis.copy()
        else:
            self.hard_X = np.vstack([self.hard_X, X_mis])
            self.hard_y = np.concatenate([self.hard_y, y_mis])
        # trim to buffer size
        if len(self.hard_y) > self.hard_buffer_size:
            keep = self.hard_buffer_size
            self.hard_X = self.hard_X[-keep:]
            self.hard_y = self.hard_y[-keep:]

    def _sample_train_mix(self, X_new, y_new):
        if len(self.hard_y) == 0 or self.replay_mix <= 0.0:
            return X_new, y_new
        n_new = len(X_new)
        n_replay = int(round(n_new * self.replay_mix))
        n_replay = min(n_replay, len(self.hard_y))
        if n_replay == 0:
            return X_new, y_new
        idx = self.rng.choice(len(self.hard_y), size=n_replay, replace=False)
        X_replay = self.hard_X[idx]
        y_replay = self.hard_y[idx]
        X_comb = np.vstack([X_new, X_replay])
        y_comb = np.concatenate([y_new, y_replay])
        return X_comb, y_comb

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self._ensure_label_map(y)
        K = len(self.classes_)
        # update non-logic state
        self.nl_state.fit(X, y)
        warm = self.nl_state.get_warm(self.D)
        # sample mix with hard replay
        X_train, y_train = self._sample_train_mix(X, y)
        # simple class-balancing heuristic
        unique, counts = np.unique(y_train, return_counts=True)
        if len(counts) > 1:
            med = int(np.median(counts))
            max_allow = max(med * 3, med + 1)
            if max_allow < len(y_train):
                keep_idx = []
                for cls in unique:
                    idxs = np.where(y_train == cls)[0]
                    if len(idxs) > max_allow:
                        keep = self.rng.choice(idxs, max_allow, replace=False)
                    else:
                        keep = idxs
                    keep_idx.extend(keep.tolist())
                X_train = X_train[keep_idx]
                y_train = y_train[keep_idx]
        # features & solve ridge regression
        Phi = self._rff(X_train)
        n, D = Phi.shape
        y_idx = np.array([self.class_to_idx[int(v)] for v in y_train], dtype=np.int32)
        Y_onehot = np.zeros((n, K), dtype=np.float32)
        Y_onehot[np.arange(n), y_idx] = 1.0
        PhiT_Phi = Phi.T @ Phi
        PhiT_Y = Phi.T @ Y_onehot
        H = PhiT_Phi + self.ridge * np.eye(D, dtype=np.float32)
        if self.alpha is None:
            if warm is not None:
                warm_exp = np.tile(warm[:, None], (1, K)).astype(np.float32)
                self.alpha = np.linalg.solve(H, PhiT_Y + 1e-3 * warm_exp)
            else:
                self.alpha = np.linalg.solve(H, PhiT_Y)
        else:
            try:
                new_alpha = np.linalg.solve(H, PhiT_Y)
            except np.linalg.LinAlgError:
                new_alpha = np.linalg.pinv(H) @ PhiT_Y
            blend = self.nl_state.blend
            self.alpha = (1.0 - blend) * self.alpha + blend * new_alpha
        # compute misclassified on incoming batch (original X,y) and add to hard buffer
        preds_orig = self.predict(X)
        mis_idx = np.where(preds_orig != y)[0]
        if len(mis_idx) > 0:
            self._add_hard_examples(X[mis_idx], y[mis_idx])
        return self

    def predict(self, X):
        if self.alpha is None:
            return np.zeros(len(X), dtype=int)
        Phi = self._rff(X)
        scores = Phi @ self.alpha
        idxs = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in idxs], dtype=int)

    def adapt_on_feedback(self, acc):
        self.window_acc.append(acc)
        if len(self.window_acc) > 5:
            self.window_acc.pop(0)
        if self.prev_acc is None:
            self.prev_acc = acc
            return
        if acc < self.prev_acc * 0.95:
            self.nl_state.blend = min(0.9, self.nl_state.blend + 0.10)
            self.nl_state.decay = max(0.75, self.nl_state.decay - 0.05)
            self.gamma = max(0.01, self.gamma * 1.08)
            # reinitialize W with slightly higher variance
            self.W = self.rng.normal(0, np.sqrt(self.gamma), self.W.shape).astype(np.float32)
            self.adapt_counter += 1
        else:
            if acc > self.prev_acc:
                self.nl_state.blend = max(0.01, self.nl_state.blend * 0.93)
                self.nl_state.decay = min(0.99, self.nl_state.decay * 1.01)
                self.gamma = max(0.005, self.gamma * 0.995)
        if np.std(self.window_acc) > 0.03:
            self.ridge = max(0.5, self.ridge * 0.9)
        else:
            self.ridge = min(200.0, self.ridge * 1.005)
        self.prev_acc = acc


# -------------------------
# Data loader (covtype smaller chunks default)
# -------------------------
def load_data(n_chunks=8, chunk_size=4000, nrows=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("Loading dataset (may take a moment)...")
    df = pd.read_csv(url, header=None, nrows=nrows or (n_chunks * chunk_size))
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int32)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all).astype(np.float32)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)


# -------------------------
# Benchmark scenario: streaming compare
# -------------------------
def scenario_streaming(chunks, all_classes,
                       D=2048, ridge=5.0, replay_mix=0.30, hard_buffer_size=2000):
    print("\n" + "=" * 80)
    print("STREAMING: SunyataV3 vs streaming-like XGBoost")
    print("=" * 80)
    n_features = chunks[0][0].shape[1]
    sunyata = SunyataV3(n_input_features=n_features, D=D, ridge=ridge,
                        hard_buffer_size=hard_buffer_size, replay_mix=replay_mix)
    results = []
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        print(f"Chunk {chunk_id:02d}/{len(chunks)} | train={len(X_train)} test={len(X_test)}")
        # Sunyata
        t0 = time.time()
        sunyata.partial_fit(X_train, y_train, classes=all_classes)
        pred = sunyata.predict(X_test)
        acc = accuracy_score(y_test, pred)
        t_s = time.time() - t0
        sunyata.adapt_on_feedback(acc)
        # XGBoost (streaming-like small rounds)
        t0 = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train({"objective": "multi:softmax", "num_class": len(all_classes),
                               "max_depth": 4, "eta": 0.25, "verbosity": 0},
                              dtrain, num_boost_round=8)
        xgb_pred = xgb_model.predict(dtest).astype(int)
        acc_x = accuracy_score(y_test, xgb_pred)
        t_x = time.time() - t0
        print(f"  Sunyata: acc={acc:.4f} t={t_s:.3f}s  |  XGB: acc={acc_x:.4f} t={t_x:.3f}s")
        results.append({
            'chunk': chunk_id,
            'sunyata_acc': acc,
            'sunyata_time': t_s,
            'xgb_acc': acc_x,
            'xgb_time': t_x
        })
    df = pd.DataFrame(results)
    print("\nFINAL SUMMARY")
    print("------------")
    print(f"Sunyata avg acc: {df['sunyata_acc'].mean():.4f} | avg time: {df['sunyata_time'].mean():.3f}s")
    print(f"XGB     avg acc: {df['xgb_acc'].mean():.4f} | avg time: {df['xgb_time'].mean():.3f}s")
    os.makedirs('benchmark_results', exist_ok=True)
    df.to_csv('benchmark_results/sunyata_v3_streaming_results.csv', index=False)
    return df


# -------------------------
# Main
# -------------------------
def main():
    print("=" * 80)
    print("SunyataV3 Streaming Benchmark")
    print("=" * 80)
    chunks, all_classes = load_data(n_chunks=8, chunk_size=4000)
    df = scenario_streaming(chunks, all_classes,
                            D=2048, ridge=5.0, replay_mix=0.30, hard_buffer_size=2000)
    print("\nSaved results to benchmark_results/sunyata_v3_streaming_results.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
