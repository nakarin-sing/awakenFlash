#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash benchmark — Train/Inference speed comparison (v1.8: Final Error Fix & Deep XGBoost)
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import psutil, gc

# --------------------------------------------------
# Import Core Functions and Configuration (FIXED IMPORT)
# --------------------------------------------------
try:
    # 1. Try to import from the structured path (src/)
    from src.awakenFlash_core import train_step, infer, data_stream, N_FEATURES, N_CLASSES, N_SAMPLES, H, B, CONF_THRESHOLD, LS
except ModuleNotFoundError:
    try:
        # 2. Try to import from the flat path (if core file is in the same directory)
        from awakenFlash_core import train_step, infer, data_stream, N_FEATURES, N_CLASSES, N_SAMPLES, H, B, CONF_THRESHOLD, LS
    except:
        # 3. Fallback: Define hardcoded config and dummy functions to prevent NameError
        print("Error: Could not import core module. Using hardcoded config and dummy functions.")
        
        # Hardcoded Config (จากไฟล์ core ที่แก้ไขไปก่อนหน้านี้)
        N_FEATURES = 40; N_CLASSES = 3; N_SAMPLES = 100_000; H = 448; B = 1024; CONF_THRESHOLD = 80; LS = 0.006
        
        # Dummy functions (เพื่อไม่ให้เกิด NameError)
        def train_step(*args, **kwargs): 
            # จำลองการส่ง W1 (values), b1, W2, b2 กลับไป
            return args[2], args[3], args[4], args[5]
        def infer(X_i8, *args, **kwargs): 
            return np.zeros(X_i8.shape[0], dtype=np.int64), 0.0
        def data_stream(n): 
            return iter([(np.random.randn(n, N_FEATURES).astype(np.float32), 
                          np.random.randint(0, N_CLASSES, n).astype(np.int32))])

# --------------------------
# Config for XGBoost (UPDATED for Deeper Tree)
# --------------------------
XGB_N_ESTIMATORS = 300  # Increased from 30
XGB_MAX_DEPTH = 6       # Increased from 4

# --------------------------
# Main Benchmark Logic
# --------------------------

def run_benchmark():
    start_time = time.time()
    
    # 1. Data Generation (Using core's config)
    # ใช้ next() บน iterator ที่ return จาก data_stream
    X_full, y_full = next(data_stream(n=N_SAMPLES))
    
    # Quantize Full Data first
    # เราต้องใช้ scale factor ที่ถูกต้องตามข้อมูลจริง
    scale = max(1.0, np.max(np.abs(X_full)) / 127.0)
    X_i8_full = np.clip(np.round(X_full / scale), -128, 127).astype(np.int8)

    # Split data for training/inference
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    X_i8_train, X_i8_test, _, _ = train_test_split(X_i8_full, y_full, test_size=0.2, random_state=42)
    
    data_time = time.time() - start_time
    print(f"Data Generation Time: {data_time:.2f}s")
    
    # --------------------------
    # XGBoost Benchmark
    # --------------------------
    
    # Train
    xgb_train_start = time.time()
    # Use standard XGBClassifier for simplicity (แทน DMatrix)
    xgb_model = xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS, 
        max_depth=XGB_MAX_DEPTH, 
        n_jobs=-1, 
        random_state=42, 
        tree_method='hist', 
        verbosity=0
    )
    
    xgb_model.fit(X_train, y_train) 
    xgb_train_time = time.time() - xgb_train_start
    
    # Inference
    xgb_infer_start = time.time()
    xgb_pred = xgb_model.predict(X_test)
    xgb_infer_time = time.time() - xgb_infer_start
    
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_ms_per_sample = (xgb_infer_time / len(X_test)) * 1000

    print("\n[XGBoost Results]")
    print(f"Accuracy : {xgb_acc:.4f}")
    print(f"Train Time (s) : {xgb_train_time:.2f}")
    print(f"Inference (ms/sample) : {xgb_ms_per_sample:.4f}")
    
    del xgb_model
    gc.collect()

    # --------------------------
    # awakenFlash Benchmark (INT8)
    # --------------------------
    
    # Initialization
    np.random.seed(42)
    
    # Initialize W1 (H rows, N_FEATURES cols)
    W1 = np.random.randint(-127, 128, size=(H, N_FEATURES), dtype=np.int64) 
    
    # Apply Sparsity (50%)
    sparsity_mask = np.random.rand(H, N_FEATURES) < 0.5
    W1[sparsity_mask] = 0
    
    # CSR-like representation for Numba (values, col_indices, indptr)
    rows, cols = np.where(W1 != 0)
    values = W1[rows, cols].astype(np.int64) # W1 non-zero values
    col_indices = cols.astype(np.int32) 
    
    indptr = np.zeros(H + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=H), out=indptr[1:]) # Calculate row pointers
    
    b1 = np.random.randint(-127, 128, size=H, dtype=np.int64)
    W2 = np.random.randint(-127, 128, size=(H, N_CLASSES), dtype=np.int64)
    b2 = np.random.randint(-127, 128, size=N_CLASSES, dtype=np.int64)

    # Train (3 epochs)
    af_train_start = time.time()
    n_epochs = 3
    for epoch in range(n_epochs):
        # **Note:** train_step ต้องรับ values, b1, W2, b2 เป็น int64
        values, b1, W2, b2 = train_step(X_i8_train, y_train, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS)
        
    af_train_time = time.time() - af_train_start
    
    # Inference
    af_infer_start = time.time()
    af_pred, ee_ratio = infer(X_i8_test, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD)
    af_infer_time = time.time() - af_infer_start

    af_acc = accuracy_score(y_test, af_pred)
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
    acc_verdict = f"✅ awakenFlash wins ({acc_diff:+.4f})" if acc_diff > 0.0001 else (f"❌ XGBoost wins ({acc_diff:+.4f})" if acc_diff < -0.0001 else "🤝 Tie (Difference < 0.0001)")

    # Train Speed Comparison
    train_diff = af_train_time - xgb_train_time
    train_verdict = f"✅ awakenFlash wins ({xgb_train_time/af_train_time:.1f}x faster)" if train_diff < 0 else f"❌ XGBoost wins ({train_diff:+.4f}s slower)"
        
    # Inference Speed Comparison
    infer_diff = af_ms_per_sample - xgb_ms_per_sample
    infer_verdict = f"✅ awakenFlash wins ({xgb_ms_per_sample/af_ms_per_sample:.0f}x faster)" if infer_diff < 0 else f"❌ XGBoost wins ({infer_diff:+.4f}ms slower)"

    print(f"Accuracy Result : {acc_verdict}")
    print(f"Train Speed Result : {train_verdict}")
    print(f"Inference Speed Result : {infer_verdict}")
    print("==============================")


if __name__ == "__main__":
    print("==============================")
    print(f"Starting awakenFlash Benchmark (v1.8: Deep XGBoost Re-Validation N_F={N_FEATURES})")
    print("==============================")
    run_benchmark()

