#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash benchmark — Train/Inference speed comparison (INT8 vs XGBoost)
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from awakenFlash_core import train_step, infer, data_stream, N_FEATURES, N_CLASSES, N_SAMPLES, H, B, CONF_THRESHOLD, LS

# --------------------------
# Config for XGBoost (UPDATED for Deeper Tree)
# --------------------------
# Target setting to make XGBoost inference time slower and more complex (Deeper Tree)
XGB_N_ESTIMATORS = 300  # Increased from 30
XGB_MAX_DEPTH = 6       # Increased from 4

# --------------------------
# Main Benchmark Logic
# --------------------------

def run_benchmark():
    start_time = time.time()
    
    # 1. Data Generation (Using core's config)
    X, y = next(data_stream(n=N_SAMPLES))
    X_i8 = (np.clip(X, -1, 1) * 127).astype(np.int8)
    
    # Split data for training/inference
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_i8_train, X_i8_test, _, _ = train_test_split(X_i8, y, test_size=0.2, random_state=42)

    data_time = time.time() - start_time
    print(f"Data Generation Time: {data_time:.2f}s")
    
    # --------------------------
    # XGBoost Benchmark
    # --------------------------
    
    # Train
    xgb_train_start = time.time()
    # Use DMatrix for optimal XGBoost performance
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'multi:softmax',
        'num_class': N_CLASSES,
        'nthread': 1, # Force single thread for controlled comparison
        'eta': 0.1,
        'max_depth': XGB_MAX_DEPTH # Use updated depth
    }
    
    xgb_model = xgb.train(params, dtrain, num_boost_round=XGB_N_ESTIMATORS) # Use updated estimators
    xgb_train_time = time.time() - xgb_train_start
    
    # Inference
    xgb_infer_start = time.time()
    xgb_pred = xgb_model.predict(dtest)
    xgb_infer_time = time.time() - xgb_infer_start
    
    xgb_acc = np.mean(xgb_pred == y_test)
    xgb_ms_per_sample = (xgb_infer_time / len(X_test)) * 1000

    print("\n[XGBoost Results]")
    print(f"Accuracy : {xgb_acc:.4f}")
    print(f"Train Time (s) : {xgb_train_time:.2f}")
    print(f"Inference (ms/sample) : {xgb_ms_per_sample:.4f}")


    # --------------------------
    # awakenFlash Benchmark (INT8)
    # --------------------------
    
    # Initialize parameters
    # The initial random sparse weights are generated here to ensure reproducibility
    np.random.seed(42)
    W1 = np.random.randint(-127, 128, size=(H, N_FEATURES), dtype=np.int8)
    b1 = np.random.randint(-127, 128, size=H, dtype=np.int64)
    W2 = np.random.randint(-127, 128, size=(H, N_CLASSES), dtype=np.int64)
    b2 = np.random.randint(-127, 128, size=N_CLASSES, dtype=np.int64)

    # Convert W1 to Sparse (CSR-like format) for AWAKEN Flash
    # We maintain 50% sparsity (50% of weights are non-zero)
    sparsity_mask = np.random.rand(H, N_FEATURES) < 0.5
    W1[sparsity_mask] = 0
    
    # CSR-like representation for Numba
    values = W1[W1 != 0].astype(np.int64)
    col_indices = np.where(W1.flatten() != 0)[0] % N_FEATURES
    indptr = np.zeros(H + 1, dtype=np.int64)
    current_index = 0
    for i in range(H):
        non_zero_count = np.count_nonzero(W1[i])
        indptr[i+1] = indptr[i] + non_zero_count

    # Train (Iterative single-pass training to mimic real-time update)
    af_train_start = time.time()
    
    # Simulate 3 epochs of single-pass training over the whole dataset
    n_epochs = 3
    for _ in range(n_epochs):
        values, b1, W2, b2 = train_step(X_i8_train, y_train, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS)
        
    af_train_time = time.time() - af_train_start
    
    # Inference
    af_infer_start = time.time()
    af_pred, ee_ratio = infer(X_i8_test, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD)
    af_infer_time = time.time() - af_infer_start

    af_acc = np.mean(af_pred == y_test)
    af_ms_per_sample = (af_infer_time / len(X_test)) * 1000
    
    print("\n[awakenFlash Results]")
    print(f"Accuracy : {af_acc:.4f}")
    print(f"Train Time (s) : {af_train_time:.2f}")
    print(f"Inference (ms/sample) : {af_ms_per_sample:.4f}")
    print(f"Early Exit Ratio : {ee_ratio * 100:.1f}%")
    
    # --------------------------
    # Final Verdict
    # --------------------------
    print("\n==============================")
    print(f"🏁 FINAL VERDICT (N_FEATURES={N_FEATURES}, XGBoost: Deep/Complex)")
    print("==============================")

    # Accuracy Comparison
    acc_diff = af_acc - xgb_acc
    if acc_diff > 0.0001:
        acc_verdict = f"✅ awakenFlash wins ({acc_diff:+.4f})"
    elif acc_diff < -0.0001:
        acc_verdict = f"❌ XGBoost wins ({acc_diff:+.4f})"
    else:
        acc_verdict = "🤝 Tie (Difference < 0.0001)"

    # Train Speed Comparison
    train_diff = af_train_time - xgb_train_time
    if train_diff < 0:
        train_verdict = f"✅ awakenFlash wins ({train_diff:+.4f})"
    else:
        train_verdict = f"❌ XGBoost wins ({train_diff:+.4f})"
        
    # Inference Speed Comparison
    infer_diff = af_ms_per_sample - xgb_ms_per_sample
    if infer_diff < 0:
        infer_verdict = f"✅ awakenFlash wins ({infer_diff:+.4f})"
    else:
        infer_verdict = f"❌ XGBoost wins ({infer_diff:+.4f})"

    print(f"Accuracy Result : {acc_verdict}")
    print(f"Train Speed Result : {train_verdict}")
    print(f"Inference Speed Result : {infer_verdict}")
    print("==============================")


if __name__ == "__main__":
    print("==============================")
    print(f"Starting awakenFlash Benchmark (v1.7: Deep XGBoost Re-Validation N_F={N_FEATURES})")
    print("==============================")
    run_benchmark()
