#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming Benchmark: OneStep vs XGBoost
Dataset ~500 MB, micro-batch online updates
Designed for GitHub Actions
MIT © 2025
"""

import numpy as np
import pandas as pd
import time
import tracemalloc
import psutil
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import requests
import os

# -------------------------
# Streaming Data Loader
# -------------------------

def download_large_csv(url, local_path="dataset.csv"):
    """Download CSV from URL if not exists"""
    if not os.path.exists(local_path):
        print(f"Downloading dataset from {url} ... (~500 MB)")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")
    return local_path

def stream_csv(file_path, chunksize=100_000):
    """Yield chunks from CSV"""
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        X = chunk.drop(columns=["target"]).values.astype(np.float32)
        y = chunk["target"].values.astype(np.int32)
        yield X, y

# -------------------------
# Memory Utilities
# -------------------------

def measure_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# -------------------------
# Streaming OneStep RLS
# -------------------------

class StreamingOneStep(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1e-3, use_poly=False, poly_degree=2):
        self.C = C
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        self.W = None
        self.n_features = None
        self.n_classes = None
    
    def partial_fit(self, X, y, classes=None):
        if self.W is None:
            # initialize weights
            self.n_features = X.shape[1] + 1  # + bias
            self.n_classes = len(classes) if classes is not None else y.max() + 1
            self.W = np.zeros((self.n_features, self.n_classes), dtype=np.float32)
        
        # Add bias
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        
        # One-hot
        y_onehot = np.eye(self.n_classes, dtype=np.float32)[y]
        
        # Update using simple RLS-like rule
        # gain = (X^T X + lambda I)^-1 X^T
        XTX = X_b.T @ X_b
        lambda_adaptive = self.C * np.trace(XTX) / XTX.shape[0]
        I = np.eye(XTX.shape[0], dtype=np.float32)
        gain = np.linalg.solve(XTX + lambda_adaptive*I, X_b.T)
        
        self.W += gain @ (y_onehot - X_b @ self.W)
        
        del XTX, I, gain, X_b, y_onehot
        gc.collect()
        return self
    
    def predict(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        logits = X_b @ self.W
        return logits.argmax(axis=1)

# -------------------------
# Benchmark Runner
# -------------------------

def benchmark_streaming(url):
    # Download dataset
    csv_file = download_large_csv(url)
    
    # Memory tracking
    mem_start = measure_memory()
    tracemalloc.start()
    
    # Streaming setup
    one_step = StreamingOneStep(C=0.01)
    X_test_all = []
    y_test_all = []
    
    t0 = time.time()
    for i, (X_chunk, y_chunk) in enumerate(stream_csv(csv_file, chunksize=100_000)):
        if i == 0:
            # take 20% as test from first chunk
            X_train, X_test, y_train, y_test = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=42)
            X_test_all.append(X_test)
            y_test_all.append(y_test)
            one_step.partial_fit(X_train, y_train, classes=np.unique(y_chunk))
        else:
            one_step.partial_fit(X_chunk, y_chunk)
        if (i+1) % 5 == 0:
            print(f"Processed { (i+1)*100_000 } rows ...")
    t_elapsed = time.time() - t0
    
    # Evaluate
    X_test_all = np.vstack(X_test_all)
    y_test_all = np.concatenate(y_test_all)
    y_pred = one_step.predict(X_test_all)
    acc = accuracy_score(y_test_all, y_pred)
    f1 = f1_score(y_test_all, y_pred, average='weighted')
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_end = measure_memory()
    
    print("="*80)
    print(f"Streaming OneStep Benchmark Complete")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"Time elapsed: {t_elapsed:.2f}s")
    print(f"Memory used (RSS): {mem_end - mem_start:.2f} MB")
    print(f"Peak memory (tracemalloc): {peak_mem/1024/1024:.2f} MB")
    print("="*80)

if __name__ == "__main__":
    # Example: large CSV URL
    url = "https://example.com/large_dataset_500MB.csv"  # <<< ใส่ URL จริง
    benchmark_streaming(url)
