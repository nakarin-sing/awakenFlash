#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NON-LOGIC STREAMING/BATCH BENCHMARK
Log-only, no file output, simple memory + metrics display
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import tracemalloc
import warnings
warnings.filterwarnings('ignore')


class MemoryTracker:
    def __init__(self):
        self.snapshots = []
        tracemalloc.start()
    
    def snapshot(self, label):
        current, peak = tracemalloc.get_traced_memory()
        self.snapshots.append({
            'label': label,
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024
        })
    
    def print_usage(self):
        print("\n📌 Memory Snapshots")
        for s in self.snapshots:
            print(f"{s['label']}: current={s['current_mb']:.2f} MB, peak={s['peak_mb']:.2f} MB")


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return acc, precision, recall, f1


def load_data(n_chunks=5, chunk_size=5000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"📦 Loading dataset from UCI ML repository...")
    
    df = pd.read_csv(url, header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values - 1
    
    print(f"   Dataset shape: {X.shape}, classes: {np.unique(y)}")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    chunks = [(X[i:i+chunk_size], y[i:i+chunk_size])
              for i in range(0, min(len(X), n_chunks*chunk_size), chunk_size)]
    
    print(f"   Created {len(chunks)} chunks of size {chunk_size}\n")
    return chunks, np.unique(y)


def scenario_streaming(chunks, classes, memory_tracker):
    print("="*60)
    print("🔄 STREAMING SCENARIO (Online learning)")
    print("="*60)
    
    sgd = SGDClassifier(loss="log_loss", max_iter=10, warm_start=True, n_jobs=-1)
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=10, warm_start=True, n_jobs=-1)
    
    first_sgd = first_pa = True
    
    xgb_window_X, xgb_window_y = [], []
    WINDOW_SIZE = 3
    
    for i, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        print(f"\nChunk {i} | Train: {len(X_train)}, Test: {len(X_test)}")
        
        # ===== SGD =====
        memory_tracker.snapshot(f"sgd_chunk{i}_start")
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X_train, y_train, classes=classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train)
        y_pred_sgd = sgd.predict(X_test)
        sgd_acc, sgd_prec, sgd_rec, sgd_f1 = compute_metrics(y_test, y_pred_sgd)
        sgd_time = time.time() - start
        memory_tracker.snapshot(f"sgd_chunk{i}_end")
        print(f"  SGD: acc={sgd_acc:.3f}, f1={sgd_f1:.3f}, t={sgd_time:.3f}s")
        
        # ===== PA =====
        memory_tracker.snapshot(f"pa_chunk{i}_start")
        start = time.time()
        if first_pa:
            pa.partial_fit(X_train, y_train, classes=classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train)
        y_pred_pa = pa.predict(X_test)
        pa_acc, pa_prec, pa_rec, pa_f1 = compute_metrics(y_test, y_pred_pa)
        pa_time = time.time() - start
        memory_tracker.snapshot(f"pa_chunk{i}_end")
        print(f"  PA:  acc={pa_acc:.3f}, f1={pa_f1:.3f}, t={pa_time:.3f}s")
        
        # ===== XGBoost Sliding Window =====
        memory_tracker.snapshot(f"xgb_chunk{i}_start")
        start = time.time()
        xgb_window_X.append(X_train)
        xgb_window_y.append(y_train)
        if len(xgb_window_X) > WINDOW_SIZE:
            xgb_window_X = xgb_window_X[-WINDOW_SIZE:]
            xgb_window_y = xgb_window_y[-WINDOW_SIZE:]
        X_xgb_train = np.vstack(xgb_window_X)
        y_xgb_train = np.concatenate(xgb_window_y)
        dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train({"objective":"multi:softmax","num_class":7,"verbosity":0}, dtrain, num_boost_round=10)
        y_pred_xgb = xgb_model.predict(dtest)
        xgb_acc, xgb_prec, xgb_rec, xgb_f1 = compute_metrics(y_test, y_pred_xgb)
        xgb_time = time.time() - start
        memory_tracker.snapshot(f"xgb_chunk{i}_end")
        print(f"  XGB: acc={xgb_acc:.3f}, f1={xgb_f1:.3f}, t={xgb_time:.3f}s")
    
    memory_tracker.print_usage()


def scenario_batch(chunks, classes, memory_tracker):
    print("\n" + "="*60)
    print("📦 BATCH SCENARIO (Full dataset)")
    print("="*60)
    
    X_all = np.vstack([c[0] for c in chunks])
    y_all = np.concatenate([c[1] for c in chunks])
    
    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # ===== SGD =====
    memory_tracker.snapshot("sgd_batch_start")
    start = time.time()
    sgd = SGDClassifier(loss="log_loss", max_iter=20, n_jobs=-1)
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    acc, prec, rec, f1 = compute_metrics(y_test, y_pred)
    print(f"SGD Batch: acc={acc:.3f}, f1={f1:.3f}, t={time.time()-start:.3f}s")
    memory_tracker.snapshot("sgd_batch_end")
    
    # ===== PA =====
    memory_tracker.snapshot("pa_batch_start")
    start = time.time()
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=20, n_jobs=-1)
    pa.fit(X_train, y_train)
    y_pred = pa.predict(X_test)
    acc, prec, rec, f1 = compute_metrics(y_test, y_pred)
    print(f"PA Batch:  acc={acc:.3f}, f1={f1:.3f}, t={time.time()-start:.3f}s")
    memory_tracker.snapshot("pa_batch_end")
    
    # ===== XGBoost =====
    memory_tracker.snapshot("xgb_batch_start")
    start = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train({"objective":"multi:softmax","num_class":7,"verbosity":0}, dtrain, num_boost_round=20)
    y_pred = xgb_model.predict(dtest)
    acc, prec, rec, f1 = compute_metrics(y_test, y_pred)
    print(f"XGB Batch: acc={acc:.3f}, f1={f1:.3f}, t={time.time()-start:.3f}s")
    memory_tracker.snapshot("xgb_batch_end")
    
    memory_tracker.print_usage()


if __name__ == "__main__":
    print("="*60)
    print("🚀 NON-LOGIC ML BENCHMARK")
    print("="*60)
    
    mem_tracker = MemoryTracker()
    chunks, classes = load_data(n_chunks=5, chunk_size=5000)
    
    scenario_streaming(chunks, classes, mem_tracker)
    scenario_batch(chunks, classes, mem_tracker)
    
    print("\n✅ BENCHMARK COMPLETE")
