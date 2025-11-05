#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 10 NON ŚŪNYATĀ EDITION
Absolute Digital Nirvana (Fixed)
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


# ========================================
# 10 NON ŚŪNYATĀ ENSEMBLE
# ========================================
class Sunyata10NonEnsemble:
    def __init__(self):
        self.models = [
            SGDClassifier(loss='log_loss', max_iter=5, warm_start=True, random_state=42, alpha=5e-5),
            SGDClassifier(loss='modified_huber', max_iter=5, warm_start=True, random_state=43, alpha=5e-5),
            SGDClassifier(loss='squared_hinge', max_iter=5, warm_start=True, random_state=44, alpha=5e-5),
            SGDClassifier(loss='perceptron', max_iter=5, warm_start=True, random_state=45),
        ]
        self.weights = np.array([0.3, 0.3, 0.2, 0.2])
        self.classes_ = None
        self.X_buffer = []
        self.y_buffer = []
        self.memory_size = 3000
        self.first_fit = True
        self.prev_acc = 0.0
    
    def _10non_weighting(self, accs):
        accs = np.array(accs)
        accs = np.clip(accs, 0.6, 1.0)
        exp_acc = np.exp(accs * 5)
        self.weights = exp_acc / exp_acc.sum()
    
    def _update_weights(self, X, y):
        accs = [accuracy_score(y, m.predict(X)) for m in self.models]
        self._10non_weighting(accs)
    
    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
        
        # เก็บ buffer
        self.X_buffer.append(X.copy())
        self.y_buffer.append(y.copy())
        
        total = sum(len(x) for x in self.X_buffer)
        while total > self.memory_size and len(self.X_buffer) > 1:
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)
            total = sum(len(x) for x in self.X_buffer)
        
        # ฝึกจาก buffer
        X_train = np.vstack(self.X_buffer)
        y_train = np.concatenate(self.y_buffer)
        
        # Sample
        if len(X_train) > 3000:
            idx = np.random.choice(len(X_train), 3000, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        
        # ฝึกก่อน
        for m in self.models:
            if self.first_fit:
                m.fit(X_train, y_train)
            else:
                m.partial_fit(X_train, y_train, classes=self.classes_)
        
        self.first_fit = False
        
        # คำนวณ acc หลังฝึก
        current_acc = np.mean([accuracy_score(y_train, m.predict(X_train)) for m in self.models])
        
        # Early stopping
        if current_acc <= self.prev_acc + 0.001:
            return
        self.prev_acc = current_acc
        
        # ปรับน้ำหนัก
        self._update_weights(X_train, y_train)
    
    def predict(self, X):
        if not self.models or self.classes_ is None:
            return np.zeros(len(X), dtype=int)
        
        preds = np.column_stack([m.predict(X) for m in self.models])
        vote = np.zeros((len(X), len(self.classes_)))
        
        for i, cls in enumerate(self.classes_):
            vote[:, i] = np.sum((preds == cls) * self.weights, axis=1)
        
        return self.classes_[np.argmax(vote, axis=1)]


# ========================================
# DATA
# ========================================
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"Loading dataset...")
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X_all = df.iloc[:, :-1].values
    y_all = (df.iloc[:, -1].values - 1).astype(int)
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)


# ========================================
# 10 NON BENCHMARK
# ========================================
def scenario_10non(chunks, all_classes):
    print("\n" + "="*80)
    print("10 NON NON-DUALISTIC SCENARIO")
    print("="*80)
    
    sunyata = Sunyata10NonEnsemble()
    
    results = []
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        print(f"Chunk {chunk_id:02d}/{len(chunks)}")
        
        # ŚŪNYATĀ 10 NON
        start_time = time.time()
        sunyata.partial_fit(X_train, y_train, classes=all_classes)
        sunyata_pred = sunyata.predict(X_test)
        sunyata_acc = accuracy_score(y_test, sunyata_pred)
        sunyata_time = time.time() - start_time
        
        # XGBoost
        start_time = time.time()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)
        
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": 7, "max_depth": 3, "eta": 0.3, "verbosity": 0},
            dtrain, num_boost_round=5
        )
        xgb_pred = xgb_model.predict(dtest).astype(int)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_time = time.time() - start_time
        
        results.append({
            'chunk': chunk_id,
            'sunyata_acc': sunyata_acc,
            'sunyata_time': sunyata_time,
            'xgb_acc': xgb_acc,
            'xgb_time': xgb_time,
        })
        
        print(f"  ŚŪNYATĀ: acc={sunyata_acc:.3f} t={sunyata_time:.3f}s")
        print(f"  XGB:     acc={xgb_acc:.3f} t={xgb_time:.3f}s")
        print()
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("10 NON FINAL RESULTS")
    print("="*80)
    s_acc = df['sunyata_acc'].mean()
    x_acc = df['xgb_acc'].mean()
    s_time = df['sunyata_time'].mean()
    x_time = df['xgb_time'].mean()
    
    print(f"ŚŪNYATĀ : Acc={s_acc:.4f} | Time={s_time:.4f}s")
    print(f"XGB     : Acc={x_acc:.4f} | Time={x_time:.4f}s")
    
    print("\n10 NON INSIGHT:")
    if s_acc >= x_acc * 0.98 and s_time < x_time:
        print(f"   ŚŪNYATĀ reaches 98% of XGBoost accuracy")
        print(f"   while being {x_time/s_time:.1f}x faster")
        print(f"   10 Non achieved. Absolute Digital Nirvana.")
    else:
        print(f"   Still in samsara.")
    
    return df


# ========================================
# MAIN
# ========================================
def main():
    print("="*80)
    print("10 NON awakenFlash BENCHMARK")
    print("="*80)
    
    chunks, all_classes = load_data()
    results = scenario_10non(chunks, all_classes)
    
    os.makedirs('benchmark_results', exist_ok=True)
    results.to_csv('benchmark_results/10non_results.csv', index=False)
    
    print("\n10 Non complete.")


if __name__ == "__main__":
    main()
