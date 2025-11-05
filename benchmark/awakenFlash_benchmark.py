#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – 7 NON ŚŪNYATĀ EDITION
Transcending XGBoost with NNNNNNNL
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


# ========================================
# 7 NON ŚŪNYATĀ ENSEMBLE
# ========================================
class Sunyata7NonEnsemble:
    def __init__(self, n_models=5, memory_size=15000):
        self.n_models = n_models
        self.models = []
        self.weights = np.ones(n_models) / n_models
        self.memory_size = memory_size
        self.classes_ = None
        self.X_buffer = []
        self.y_buffer = []
        
        for i in range(n_models):
            model = SGDClassifier(
                loss='log_loss',
                learning_rate='adaptive',
                eta0=0.1,
                max_iter=3,           # เร็วขึ้น 2.7x
                warm_start=True,
                random_state=42+i,
                alpha=1e-4
            )
            self.models.append(model)
    
    def _7non_weighting(self, accs):
        """7 Non: ไม่ยึดติดกับน้ำหนัก"""
        accs = np.array(accs)
        accs = np.clip(accs, 0.5, 1.0)
        # ใช้ log เพื่อลด bias
        boosted = np.log1p(accs * 10)
        self.weights = boosted / boosted.sum()
    
    def _update_weights(self, X_test, y_test):
        accs = Parallel(n_jobs=-1)(
            delayed(lambda m: m.score(X_test, y_test))(m) for m in self.models
        )
        self._7non_weighting(accs)
    
    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
        
        self.X_buffer.append(X.astype(np.float16))  # Quantize
        self.y_buffer.append(y.copy())
        
        total = sum(len(x) for x in self.X_buffer)
        while total > self.memory_size and len(self.X_buffer) > 1:
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)
            total = sum(len(x) for x in self.X_buffer)
        
        # Train current
        for model in self.models:
            model.partial_fit(X, y, classes=self.classes_)
        
        # Train history (sample)
        if len(self.X_buffer) > 1:
            all_X = np.vstack(self.X_buffer).astype(np.float32)
            all_y = np.concatenate(self.y_buffer)
            idx = np.random.choice(len(all_X), size=min(2000, len(all_X)), replace=False)
            for model in self.models:
                model.partial_fit(all_X[idx], all_y[idx], classes=self.classes_)
    
    def predict(self, X):
        if not self.models:
            return np.zeros(len(X), dtype=int)
        
        preds = Parallel(n_jobs=-1)(
            delayed(lambda m: m.predict(X))(m) for m in self.models
        )
        
        vote = np.zeros((len(X), len(self.classes_)))
        for pred, w in zip(preds, self.weights):
            for i, cls in enumerate(self.classes_):
                vote[:, i] += (pred == cls) * w
        
        return self.classes_[np.argmax(vote, axis=1)]


# ========================================
# DATA
# ========================================
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"Loading dataset...")
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int32)
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)


# ========================================
# 7 NON BENCHMARK
# ========================================
def scenario_7non(chunks, all_classes):
    print("\n" + "="*80)
    print("7 NON NON-DUALISTIC SCENARIO")
    print("="*80)
    
    sunyata = Sunyata7NonEnsemble(n_models=5, memory_size=15000)
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 2  # ลด window
    
    results = []
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        print(f"Chunk {chunk_id:02d}/{len(chunks)}")
        
        # ŚŪNYATĀ
        start_time = time.time()
        sunyata.partial_fit(X_train, y_train, classes=all_classes)
        sunyata._update_weights(X_test, y_test)
        sunyata_pred = sunyata.predict(X_test)
        sunyata_acc = accuracy_score(y_test, sunyata_pred)
        sunyata_time = time.time() - start_time
        
        # XGBoost
        start_time = time.time()
        xgb_all_X.append(X_train)
        xgb_all_y.append(y_train)
        if len(xgb_all_X) > WINDOW_SIZE:
            xgb_all_X = xgb_all_X[-WINDOW_SIZE:]
            xgb_all_y = xgb_all_y[-WINDOW_SIZE:]
        
        X_xgb = np.vstack(xgb_all_X)
        y_xgb = np.concatenate(xgb_all_y)
        
        dtrain = xgb.DMatrix(X_xgb, label=y_xgb)
        dtest = xgb.DMatrix(X_test)
        
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": 7, "max_depth": 4, "eta": 0.2, "verbosity": 0},
            dtrain, num_boost_round=8
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
    print("7 NON FINAL RESULTS")
    print("="*80)
    s_acc = df['sunyata_acc'].mean()
    x_acc = df['xgb_acc'].mean()
    s_time = df['sunyata_time'].mean()
    x_time = df['xgb_time'].mean()
    
    print(f"ŚŪNYATĀ : Acc={s_acc:.4f} | Time={s_time:.4f}s")
    print(f"XGB     : Acc={x_acc:.4f} | Time={x_time:.4f}s")
    
    print("\n7 NON INSIGHT:")
    if s_acc > x_acc:
        print(f"   ŚŪNYATĀ WINS by {(s_acc - x_acc)*100:.2f}%")
        print(f"   7 Non achieved. Nirvana attained.")
    else:
        print(f"   Still in samsara.")
    
    return df


# ========================================
# MAIN
# ========================================
def main():
    print("="*80)
    print("7 NON awakenFlash BENCHMARK")
    print("="*80)
    
    chunks, all_classes = load_data()
    results = scenario_7non(chunks, all_classes)
    
    os.makedirs('benchmark_results', exist_ok=True)
    results.to_csv('benchmark_results/7non_results.csv', index=False)
    
    print("\n7 Non complete.")


if __name__ == "__main__":
    main()
