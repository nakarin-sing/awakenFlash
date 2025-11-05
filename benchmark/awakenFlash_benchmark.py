#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 8 NON ŚŪNYATĀ EDITION
Absolute Transcendence of XGBoost
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
# 8 NON ŚŪNYATĀ ENSEMBLE
# ========================================
class Sunyata8NonEnsemble:
    def __init__(self):
        self.models = [
            SGDClassifier(loss='log_loss', max_iter=1, warm_start=True, random_state=42),
            SGDClassifier(loss='modified_huber', max_iter=1, warm_start=True, random_state=43),
            SGDClassifier(loss='squared_hinge', max_iter=1, warm_start=True, random_state=44),
        ]
        self.weights = np.array([0.4, 0.3, 0.3])
        self.classes_ = None
        self.X_recent = None
        self.y_recent = None
    
    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
            for m in self.models:
                m.fit(X[:1], y[:1])  # Warm up
        
        # 8 Non: ไม่เก็บ history, ฝึกแค่ current + recent
        if self.X_recent is not None:
            X_train = np.vstack([self.X_recent, X])
            y_train = np.concatenate([self.y_recent, y])
        else:
            X_train, y_train = X, y
        
        # Sample 3000 ตัวอย่าง
        if len(X_train) > 3000:
            idx = np.random.choice(len(X_train), 3000, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        
        # ฝึกแค่ 1 iteration
        for m in self.models:
            m.partial_fit(X_train, y_train, classes=self.classes_)
        
        # เก็บ recent
        self.X_recent = X[-1000:].copy()
        self.y_recent = y[-1000:].copy()
    
    def predict(self, X):
        if not self.models:
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
# 8 NON BENCHMARK
# ========================================
def scenario_8non(chunks, all_classes):
    print("\n" + "="*80)
    print("8 NON NON-DUALISTIC SCENARIO")
    print("="*80)
    
    sunyata = Sunyata8NonEnsemble()
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 1  # XGBoost ฝึกแค่ chunk ล่าสุด
    
    results = []
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        print(f"Chunk {chunk_id:02d}/{len(chunks)}")
        
        # ŚŪNYATĀ 8 NON
        start_time = time.time()
        sunyata.partial_fit(X_train, y_train, classes=all_classes)
        sunyata_pred = sunyata.predict(X_test)
        sunyata_acc = accuracy_score(y_test, sunyata_pred)
        sunyata_time = time.time() - start_time
        
        # XGBoost
        start_time = time.time()
        xgb_all_X = [X_train]
        xgb_all_y = [y_train]
        
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
    print("8 NON FINAL RESULTS")
    print("="*80)
    s_acc = df['sunyata_acc'].mean()
    x_acc = df['xgb_acc'].mean()
    s_time = df['sunyata_time'].mean()
    x_time = df['xgb_time'].mean()
    
    print(f"ŚŪNYATĀ : Acc={s_acc:.4f} | Time={s_time:.4f}s")
    print(f"XGB     : Acc={x_acc:.4f} | Time={x_time:.4f}s")
    
    print("\n8 NON INSIGHT:")
    if s_acc > x_acc and s_time < x_time * 1.5:
        print(f"   ŚŪNYATĀ WINS by {(s_acc - x_acc)*100:.2f}% accuracy")
        print(f"   and {x_time/s_time:.1f}x faster")
        print(f"   8 Non achieved. Absolute Nirvana.")
    else:
        print(f"   Still in samsara.")
    
    return df


# ========================================
# MAIN
# ========================================
def main():
    print("="*80)
    print("8 NON awakenFlash BENCHMARK")
    print("="*80)
    
    chunks, all_classes = load_data()
    results = scenario_8non(chunks, all_classes)
    
    os.makedirs('benchmark_results', exist_ok=True)
    results.to_csv('benchmark_results/8non_results.csv', index=False)
    
    print("\n8 Non complete.")


if __name__ == "__main__":
    main()
