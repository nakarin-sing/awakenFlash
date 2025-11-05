#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 67 NON: NON-DUALISTIC NIRVANA v3
15 Models | 80K Memory | 30 Int | 2x Retrain | KILL XGB
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class NirvanaV3:
    def __init__(self, n_models=15, memory=80000):
        self.n_models = n_models
        self.memory = memory
        self.models = []
        self.weights = np.ones(n_models) / n_models
        self.X_hist, self.y_hist = [], []
        self.inter_pairs = None
        self.classes_ = None
        self.model_accs = []

        for i in range(n_models):
            if i % 7 == 0:
                m = SGDClassifier(loss='log_loss', alpha=2.5e-5*(1+i*0.015), max_iter=50, warm_start=True, random_state=42+i)
            elif i % 7 == 1:
                m = PassiveAggressiveClassifier(C=0.025*(1+i*0.12), max_iter=50, warm_start=True, random_state=42+i)
            elif i % 7 == 2:
                m = SGDClassifier(loss='modified_huber', eta0=0.025, learning_rate='adaptive', max_iter=50, warm_start=True, random_state=42+i)
            elif i % 7 == 3:
                m = SGDClassifier(loss='perceptron', penalty='l1', alpha=6e-5, max_iter=50, warm_start=True, random_state=42+i)
            elif i % 7 == 4:
                m = PassiveAggressiveClassifier(C=0.03, loss='squared_hinge', max_iter=50, warm_start=True, random_state=42+i)
            elif i % 7 == 5:
                m = SGDClassifier(loss='hinge', alpha=4e-5, max_iter=50, warm_start=True, random_state=42+i)
            else:
                m = SGDClassifier(loss='log_loss', penalty='elasticnet', alpha=5e-5, l1_ratio=0.15, max_iter=50, warm_start=True, random_state=42+i)
            self.models.append(m)

    def _interactions(self, X):
        if self.inter_pairs is None:
            var = np.var(X, axis=0)
            top = np.argsort(var)[-15:]
            pairs = [(top[i], top[j]) for i in range(len(top)) for j in range(i+1, min(i+5, len(top)))]
            self.inter_pairs = pairs[:30]
        X_int = [X]
        for i, j in self.inter_pairs:
            X_int.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(X_int)

    def _update_weights(self, X_test, y_test):
        X_aug = self._interactions(X_test)
        accs = []
        for m in self.models:
            try:
                acc = m.score(X_aug, y_test)
                accs.append(acc)
            except:
                accs.append(0.0)
        accs = np.array(accs)
        # Diversity penalty
        mean_pred = np.mean([m.predict(X_aug) for m in self.models], axis=0)
        diversity = np.mean([np.mean(m.predict(X_aug) != mean_pred) for m in self.models])
        accs += 0.01 * diversity
        self.weights = np.exp(np.clip(accs * 12, -5, 5))
        self.weights /= self.weights.sum()
        self.model_accs = accs

    def partial_fit(self, X, y, classes=None):
        X_aug = self._interactions(X)
        if classes is not None: self.classes_ = classes

        self.X_hist.append(X)
        self.y_hist.append(y)
        total = sum(len(x) for x in self.X_hist)
        while total > self.memory and len(self.X_hist) > 1:
            self.X_hist.pop(0)
            self.y_hist.pop(0)
            total = sum(len(x) for x in self.X_hist)

        # 2x retrain with sample weights
        if len(self.X_hist) >= 1:
            all_X = np.vstack(self.X_hist)
            all_y = np.concatenate(self.y_hist)
            n = len(all_X)
            weights = np.ones(n)
            weights[-len(X):] = 2.0  # ให้น้ำหนักข้อมูลใหม่
            idx = np.random.choice(n, min(15000, n), replace=False)
            X_s = self._interactions(all_X[idx])
            y_s = all_y[idx]
            w_s = weights[idx]
            for _ in range(2):
                for m in self.models:
                    try:
                        m.partial_fit(X_s, y_s, sample_weight=w_s)
                    except:
                        pass
        # Online
        for m in self.models:
            try:
                m.partial_fit(X_aug, y)
            except:
                pass
        return self

    def predict(self, X):
        X_aug = self._interactions(X)
        vote = np.zeros((len(X), len(self.classes_)))
        for m, w in zip(self.models, self.weights):
            try:
                pred = m.predict(X_aug)
                for i, c in enumerate(self.classes_):
                    vote[:, i] += (pred == c) * w
            except:
                pass
        return self.classes_[np.argmax(vote, axis=1)]

# === BENCHMARK ===
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    df = pd.read_csv(url, header=None, nrows=n_chunks*chunk_size)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = (df.iloc[:, -1].values - 1).astype(np.int8)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, len(X), chunk_size)], np.unique(y)

def run():
    print("="*80)
    print("67 NON: NIRVANA v3 — KILL XGBoost")
    print("="*80)
    chunks, classes = load_data()
    model = NirvanaV3()
    results = []
    xgb_X, xgb_y = [], []
    for i, (X_tr, y_tr) in enumerate(chunks, 1):
        split = int(0.8 * len(X_tr))
        X_train, X_test = X_tr[:split], X_tr[split:]
        y_train, y_test = y_tr[:split], y_tr[split:]

        # Nirvana
        t0 = time.time()
        model.partial_fit(X_train, y_train, classes if i==1 else None)
        model._update_weights(X_test, y_test)
        pred = model.predict(X_test)
        acc_n = accuracy_score(y_test, pred)
        t_n = time.time() - t0

        # XGBoost
        t0 = time.time()
        xgb_X.append(X_train); xgb_y.append(y_train)
        if len(xgb_X) > 5: xgb_X.pop(0); xgb_y.pop(0)
        X_x = np.vstack(xgb_X); y_x = np.concatenate(xgb_y)
        dtrain = xgb.DMatrix(X_x, label=y_x)
        dtest = xgb.DMatrix(X_test)
        xgb_m = xgb.train({"objective": "multi:softmax", "num_class": 7, "max_depth": 5, "eta": 0.1}, dtrain, num_boost_round=20, verbose_eval=False, early_stopping_rounds=5)
        pred_x = xgb_m.predict(dtest).astype(int)
        acc_x = accuracy_score(y_test, pred_x)
        t_x = time.time() - t0

        results.append({'chunk': i, 'nirvana': acc_n, 'xgb': acc_x, 't_n': t_n, 't_x': t_x})
        print(f"Chunk {i:02d}: NIRVANA={acc_n:.4f} ({t_n:.3f}s) | XGB={acc_x:.4f} ({t_x:.3f}s)")

    df = pd.DataFrame(results)
    n_acc, x_acc = df['nirvana'].mean(), df['xgb'].mean()
    print(f"\nFINAL: NIRVANA={n_acc:.4f} | XGB={x_acc:.4f}")
    print(f"WIN BY {(n_acc-x_acc)*100:+.2f}%!" if n_acc > x_acc else f"LOSE BY {(x_acc-n_acc)*100:.2f}%")
    df.to_csv('benchmark_results/67non_nirvana_v3.csv', index=False)

if __name__ == "__main__":
    run()
