#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming Hybrid: AbsoluteNon + OneStepRLS + TemporalBaseModels + XGBoost
Dataset: Covertype
"""

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
import xgboost as xgb
from sklearn.datasets import fetch_covtype

# ------- AbsoluteNon Transformation -------
def absolute_non(X, n=5, alpha=0.7, s=0.8):
    X = np.asarray(X, dtype=np.float64)
    X_folded = X.copy()
    for _ in range(n):
        X_folded = alpha * (0.5 - np.abs(X_folded - 0.5)) + (1 - alpha) * X_folded
    return X_folded * s

# ------- OneStep / RLS Closed‑Form Predictor -------
class OneStepRLS:
    def __init__(self, lam=1e-2):
        self.lam = lam
        self.X_train = None
        self.Y_train = None
        self.alpha = None

    def partial_fit(self, X, Y):
        if self.X_train is None:
            self.X_train = X.copy()
            self.Y_train = Y.copy()
        else:
            self.X_train = np.vstack([self.X_train, X])
            self.Y_train = np.vstack([self.Y_train, Y])
        K = self.X_train.dot(self.X_train.T)
        n = K.shape[0]
        self.alpha = np.linalg.solve(K + self.lam * np.eye(n), self.Y_train)

    def predict(self, X):
        return (X.dot(self.X_train.T)).dot(self.alpha)

# ------- Hybrid Streaming Model -------
class HybridStreamer:
    def __init__(self, window_size=5, memory_size=50000):
        self.scaler = StandardScaler()
        self.non_n = 5
        self.non_alpha = 0.7
        self.one_step = OneStepRLS(lam=1e-2)
        self.temporal_models = [
            SGDClassifier(loss='log_loss', max_iter=10, warm_start=True),
            PassiveAggressiveClassifier(C=0.01, max_iter=10, warm_start=True)
        ]
        self.temporal_weights = np.ones(len(self.temporal_models)) / len(self.temporal_models)
        self.window_size = window_size
        self.memory_size = memory_size
        self.X_window = []
        self.y_window = []

    def partial_fit(self, X_batch, y_batch, classes):
        X_scaled = self.scaler.fit_transform(X_batch)
        X_folded = absolute_non(X_scaled, n=self.non_n, alpha=self.non_alpha)

        # one‑step
        Y_onehot = np.eye(len(classes))[y_batch]
        self.one_step.partial_fit(X_folded, Y_onehot)

        # temporal models
        for mdl in self.temporal_models:
            try:
                mdl.partial_fit(X_folded, y_batch, classes=classes)
            except:
                pass

        # memory window
        self.X_window.append(X_folded)
        self.y_window.append(y_batch)
        if sum(len(x) for x in self.X_window) > self.memory_size:
            self.X_window.pop(0)
            self.y_window.pop(0)

    def predict(self, X_batch):
        X_scaled = self.scaler.transform(X_batch)
        X_folded = absolute_non(X_scaled, n=self.non_n, alpha=self.non_alpha)

        pred_one = self.one_step.predict(X_folded)
        pred_one_labels = np.argmax(pred_one, axis=1)

        preds_temp = []
        for mdl in self.temporal_models:
            try:
                preds_temp.append(mdl.predict(X_folded))
            except:
                preds_temp.append(np.zeros(X_folded.shape[0], dtype=int))

        all_preds = np.vstack([pred_one_labels] + preds_temp)
        weights = np.concatenate(([0.5], self.temporal_weights))
        weighted = np.apply_along_axis(lambda row: np.bincount(row, weights=weights, minlength=len(np.unique(pred_one_labels))), axis=0, arr=all_preds.T)
        return np.argmax(weighted, axis=1)

# ------- Benchmark Streaming -------
def streaming_benchmark(chunk_size=10000, n_chunks=10):
    data = fetch_covtype(return_X_y=True)
    X_all, y_all = data
    classes = np.unique(y_all)
    streamer = HybridStreamer(window_size=5, memory_size=50000)
    accs = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        X_batch = X_all[start:end]
        y_batch = y_all[start:end]

        streamer.partial_fit(X_batch, y_batch, classes)
        pred = streamer.predict(X_batch)
        acc = accuracy_score(y_batch, pred)
        accs.append(acc)
        print(f"Chunk {i+1:02d}/{n_chunks} Accuracy = {acc:.4f}")

    print(f"\n📊 Average Accuracy over {n_chunks} chunks: {np.mean(accs):.4f}")

if __name__ == "__main__":
    streaming_benchmark(chunk_size=10000, n_chunks=10)
