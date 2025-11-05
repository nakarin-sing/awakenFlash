#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark_v8non_streaming_fixed.py
Sunyata 8-Non streaming-improved vs streaming XGBoost
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


class Sunyata8NonEnsemble:
    """
    - 3 base SGD classifiers (different losses) trained with partial_fit
    - online meta-learner (SGD) that stacks base predictions
    - adaptive recent-memory sampling with sample_weight emphasizing recent data
    - light stabilization of ensemble weights
    """
    def __init__(self, recent_keep=2000, sample_cap=3000):
        self.models = [
            SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=42),
            SGDClassifier(loss='modified_huber', max_iter=1, warm_start=True, random_state=43),
            SGDClassifier(loss='squared_hinge', max_iter=1, warm_start=True, random_state=44),
        ]
        self.meta = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=99)
        self.weights = np.array([0.4, 0.3, 0.3], dtype=float)
        self.classes_ = None
        self.X_recent = None
        self.y_recent = None
        self.first_fit = True
        self.recent_keep = recent_keep
        self.sample_cap = sample_cap

    def _stabilize_weights(self):
        # small random smoothing toward dirichlet to avoid collapse
        alpha = 0.05
        noise = np.random.dirichlet(np.ones(len(self.weights))) * alpha
        self.weights = (1 - alpha) * self.weights + noise
        self.weights = np.maximum(self.weights, 1e-6)
        self.weights = self.weights / self.weights.sum()

    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = np.array(classes)

        # build train set = recent + current, with emphasis on recent
        if self.X_recent is not None:
            X_train = np.vstack([self.X_recent, X])
            y_train = np.concatenate([self.y_recent, y])
            # sample weights: recent portion gets higher weight
            recent_len = len(self.X_recent)
            cur_len = len(X)
            # emphasize current more (alpha)
            alpha = 0.7
            w = np.concatenate([np.ones(recent_len) * (1 - alpha), np.ones(cur_len) * alpha])
        else:
            X_train, y_train = X, y
            w = np.ones(len(y_train))

        # cap sample size to sample_cap for speed
        if len(X_train) > self.sample_cap:
            idx = np.random.choice(len(X_train), self.sample_cap, replace=False)
            X_train, y_train, w = X_train[idx], y_train[idx], w[idx]

        # train base models
        for m in self.models:
            if self.first_fit:
                # initial fit must know classes
                m.fit(X_train, y_train)
            else:
                m.partial_fit(X_train, y_train, classes=self.classes_, sample_weight=w)

        # train/update meta learner on base predictions (online stacking)
        preds = np.column_stack([m.predict(X_train) for m in self.models])
        # convert preds to integer labels starting 0..C-1 for meta input (use as features)
        # meta is trained to predict y_train from preds (as categorical features)
        # encode preds into one-hot per base to give meta richer signal
        meta_X = []
        for col in range(preds.shape[1]):
            # one-hot encode this base's prediction into C columns
            one_hot = np.zeros((len(preds), len(self.classes_)), dtype=int)
            for i, cls in enumerate(self.classes_):
                one_hot[:, i] = (preds[:, col] == cls).astype(int)
            meta_X.append(one_hot)
        meta_X = np.hstack(meta_X)

        if self.first_fit:
            # initial meta fit must have classes
            self.meta.fit(meta_X, y_train)
        else:
            self.meta.partial_fit(meta_X, y_train, classes=self.classes_)

        # update recent memory (keep last recent_keep from combined X_train)
        combined_X = np.vstack([self.X_recent, X]) if self.X_recent is not None else X
        combined_y = np.concatenate([self.y_recent, y]) if self.y_recent is not None else y
        if len(combined_X) > self.recent_keep:
            combined_X = combined_X[-self.recent_keep:]
            combined_y = combined_y[-self.recent_keep:]
        self.X_recent = combined_X.copy()
        self.y_recent = combined_y.copy()

        # stabilize ensemble weights slightly
        self._stabilize_weights()
        self.first_fit = False

    def predict(self, X):
        if self.classes_ is None:
            return np.zeros(len(X), dtype=int)
        base_preds = np.column_stack([m.predict(X) for m in self.models])
        # create meta features same as in training
        meta_X = []
        for col in range(base_preds.shape[1]):
            one_hot = np.zeros((len(base_preds), len(self.classes_)), dtype=int)
            for i, cls in enumerate(self.classes_):
                one_hot[:, i] = (base_preds[:, col] == cls).astype(int)
            meta_X.append(one_hot)
        meta_X = np.hstack(meta_X)
        meta_pred = self.meta.predict(meta_X)
        return meta_pred.astype(int)


# ---------- data loader ----------
def load_data(n_chunks=10, chunk_size=10000, nrows=None):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("Loading dataset (may take a moment)...")
    df = pd.read_csv(url, header=None, nrows=nrows)
    X_all = df.iloc[:, :-1].values
    y_all = (df.iloc[:, -1].values - 1).astype(int)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, min(len(X_all), n_chunks * chunk_size), chunk_size)]
    classes = np.unique(y_all)
    return chunks[:n_chunks], classes


# ---------- benchmark scenario ----------
def scenario_streaming(chunks, classes):
    print("\n" + "="*60)
    print("STREAMING: Sunyata 8-Non (improved) vs streaming-like XGBoost")
    print("="*60)
    sunyata = Sunyata8NonEnsemble(recent_keep=2500, sample_cap=3000)
    results = []

    # xgb window param (only recent samples)
    xgb_window = 2000
    xgb_params = {"objective": "multi:softmax", "num_class": len(classes),
                  "max_depth": 4, "eta": 0.2, "verbosity": 0}

    for idx, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        print(f"Chunk {idx}/{len(chunks)} | train={len(X_train)} test={len(X_test)}")

        # Sunyata (online)
        t0 = time.time()
        sunyata.partial_fit(X_train, y_train, classes=classes)
        pred_s = sunyata.predict(X_test)
        t_s = time.time() - t0
        acc_s = accuracy_score(y_test, pred_s)

        # XGBoost streaming-like: train on window of most recent samples (here: from chunk only)
        # to be fair, we let XGB use only the last xgb_window samples from current+recent,
        # but we simulate recent by taking last portion of X_train itself (since no global store).
        # In real streaming you'd maintain a separate buffer; here we emulate limited memory.
        t0 = time.time()
        # pick window from end of X_train
        win = min(len(X_train), xgb_window)
        X_xgb_train = X_train[-win:]
        y_xgb_train = y_train[-win:]
        dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
        dtest = xgb.DMatrix(X_test)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=4)
        pred_x = xgb_model.predict(dtest).astype(int)
        t_x = time.time() - t0
        acc_x = accuracy_score(y_test, pred_x)

        results.append({
            'chunk': idx,
            'sunyata_acc': acc_s,
            'sunyata_time': t_s,
            'xgb_acc': acc_x,
            'xgb_time': t_x
        })

        print(f"  Sunyata: acc={acc_s:.4f} time={t_s:.3f}s")
        print(f"  XGB    : acc={acc_x:.4f} time={t_x:.3f}s")
        print("-"*40)

    df = pd.DataFrame(results)
    print("\nFINAL SUMMARY")
    print("------------")
    print(f"Sunyata avg acc: {df['sunyata_acc'].mean():.4f} | avg time: {df['sunyata_time'].mean():.3f}s")
    print(f"XGB     avg acc: {df['xgb_acc'].mean():.4f} | avg time: {df['xgb_time'].mean():.3f}s")
    if df['sunyata_acc'].mean() > df['xgb_acc'].mean():
        print("=> Sunyata (Non-Logic) WINS streaming comparison.")
    else:
        print("=> XGBoost still wins. Tweak memory/weights and try again.")

    return df


def main():
    # keep nrows smaller for CI speed if needed; set nrows=None for full chunks
    chunks, classes = load_data(n_chunks=8, chunk_size=5000, nrows=8*5000)
    df = scenario_streaming(chunks, classes)
    os.makedirs('benchmark_results', exist_ok=True)
    df.to_csv('benchmark_results/8non_streaming_results.csv', index=False)
    print("\nSaved results to benchmark_results/8non_streaming_results.csv")


if __name__ == "__main__":
    main()
