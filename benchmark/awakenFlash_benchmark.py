#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Honest awakenFlash vs XGBoost benchmark (improved)
- standardize features before quantize
- val-split + simple early-stop
- CI mode (fast) when env CI=true
"""

import os, time, gc
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import psutil

# import core functions
from src.awakenFlash_core import train_step, infer, data_stream  # data_stream optional
from src.awakenFlash_core import _lut_softmax_probs_int64 as _unused_lut  # ensure module compiled
from src.awakenFlash_core import N_FEATURES as CORE_FEATURES

np.random.seed(42)

# CI mode: smaller / faster
CI_MODE = os.getenv("CI") == "true"
N_SAMPLES = 100_000 if CI_MODE else 100_000_000
EPOCHS = 2 if CI_MODE else 3
B = 1024
H = 128 if CI_MODE else 256
CONF_THRESHOLD = 80
LS = 0.006

print(f"\n[AWAKENFLASH benchmark] MODE: {'CI' if CI_MODE else 'FULL'} | N_SAMPLES = {N_SAMPLES:,}")

proc = psutil.Process()

# ---------------------------
# Create streaming dataset (use core.data_stream if available)
# we'll build a moderate train+test for CI
# ---------------------------
def make_dataset(n_samples, features=32, classes=3, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, features)).astype(np.float32)
    if features >= 16:
        X = np.hstack([X, X[:, :8] * X[:, 8:16]])
    y = rng.integers(0, classes, n_samples, dtype=np.int64)
    return X, y

# for CI, use smaller test set
TEST_N = 10000 if CI_MODE else 100000

# ---------------------------
# Prepare a real dataset (synthetic but standardized)
# ---------------------------
X_all, y_all = make_dataset(N_SAMPLES + TEST_N, features=32, classes=3, seed=42)
# split
X_train = X_all[:N_SAMPLES]
y_train = y_all[:N_SAMPLES]
X_test = X_all[N_SAMPLES:N_SAMPLES + TEST_N]
y_test = y_all[N_SAMPLES:N_SAMPLES + TEST_N]
del X_all, y_all

# standardize features (fit on train)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

# keep small val split for early-stop
val_split = min(5000, int(0.05 * N_SAMPLES))
if val_split > 0:
    X_val = X_train_s[:val_split]
    y_val = y_train[:val_split]
    X_train_s = X_train_s[val_split:]
    y_train = y_train[val_split:]

# ---------------------------
# XGBoost baseline
# ---------------------------
print("\n[1] XGBoost training ...")
ram_before = proc.memory_info().rss / 1e6
t0 = time.time()
xgb = XGBClassifier(n_estimators=30 if CI_MODE else 300,
                    max_depth=4 if CI_MODE else 6,
                    n_jobs=-1, random_state=42, tree_method='hist', verbosity=0)
xgb.fit(X_train_s, y_train)
xgb_train_time = time.time() - t0
xgb_ram = proc.memory_info().rss / 1e6 - ram_before

t0 = time.time()
xgb_pred = xgb.predict(X_test_s)
xgb_inf_ms = (time.time() - t0) / len(X_test_s) * 1000
xgb_acc = accuracy_score(y_test, xgb_pred)
del xgb_pred
gc.collect()

print(f"XGBoost acc={xgb_acc:.4f} train_s={xgb_train_time:.2f}s inf_ms={xgb_inf_ms:.4f}")

# ---------------------------
# awakenFlash: prepare INT8 quantized inputs + sparse init
# ---------------------------
print("\n[2] awakenFlash training ...")
ram_before_flash = proc.memory_info().rss / 1e6

# quantize scale based on train set
scale = max(1.0, np.max(np.abs(X_train_s)) / 127.0)
X_train_i8 = np.clip(np.round(X_train_s / scale), -128, 127).astype(np.int8)
X_test_i8 = np.clip(np.round(X_test_s / scale), -128, 127).astype(np.int8)
if val_split > 0:
    X_val_i8 = np.clip(np.round(X_val / scale), -128, 127).astype(np.int8)

# sparse init
INPUT_DIM = X_train_i8.shape[1]
mask = np.random.rand(INPUT_DIM, H) < 0.70
rows, cols = np.where(mask)
values = np.random.randint(-4, 5, size=rows.shape[0]).astype(np.int8)
col_indices = cols.astype(np.int32)
indptr = np.zeros(H + 1, np.int32)
np.cumsum(np.bincount(rows, minlength=H), out=indptr[1:])
b1 = np.zeros(H, np.int32)
W2 = np.random.randint(-4, 5, (H, 3), np.int8)
b2 = np.zeros(3, np.int32)

# prepare LUT (uint8)
lut = np.array([
    1,1,1,1,1,2,2,2,2,3,3,3,4,4,5,6,7,8,9,10,
    11,13,15,17,19,22,25,28,32,36,41,46,52,59,67,76,
    86,97,110,124,140,158,179,202,228,255
] + [255]*88, dtype=np.uint8)[:128]

# train loop (epochs)
t0 = time.time()
best_val_acc = -1.0
patience = 0
for epoch in range(EPOCHS):
    print(f"  Epoch {epoch+1}/{EPOCHS}")
    # simple batch training across the whole train set
    # we call train_step once with full train batch (train_step parallelizes by internal batch B)
    values, b1, W2, b2 = train_step(X_train_i8, y_train, values, col_indices, indptr, b1, W2, b2,
                                   B, H, CONF_THRESHOLD, LS, lut)
    # quick val check if available
    if val_split > 0:
        preds_val, _ = infer(X_val_i8, values, col_indices, indptr, b1, W2, b2, lut, CONF_THRESHOLD)
        val_acc = accuracy_score(y_val, preds_val)
        print(f"    val_acc={val_acc:.4f}")
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                print("    early stopping (val not improving)")
                break

flash_time = time.time() - t0
flash_ram = proc.memory_info().rss / 1e6 - ram_before_flash

# inference (warm-up)
for _ in range(5):
    _ = infer(X_test_i8[:1], values, col_indices, indptr, b1, W2, b2, lut, CONF_THRESHOLD)
t0 = time.time()
preds, ee_ratio = infer(X_test_i8, values, col_indices, indptr, b1, W2, b2, lut, CONF_THRESHOLD)
flash_inf_ms = (time.time() - t0) / len(X_test_i8) * 1000
flash_acc = accuracy_score(y_test, preds)
model_kb = (values.nbytes + col_indices.nbytes + indptr.nbytes + b1.nbytes + b2.nbytes + W2.nbytes + 4) / 1024

# ---------------------------
# Report
# ---------------------------
print("\n" + "="*100)
print("AWAKENFLASH vs XGBoost — honest results")
print("="*100)
print(f"{'Metric':<25} {'XGBoost':>15} {'awakenFlash':>15}")
print("-"*100)
print(f"{'Accuracy':<25} {xgb_acc:>15.4f} {flash_acc:>15.4f}")
print(f"{'Train Time (s)':<25} {xgb_train_time:>15.2f} {flash_time:>15.2f}")
print(f"{'Inference (ms/sample)':<25} {xgb_inf_ms:>15.4f} {flash_inf_ms:>15.4f}")
print(f"{'Early Exit (%)':<25} {'0%':>15} {ee_ratio*100:>14.1f}%")
print(f"{'RAM delta (MB)':<25} {xgb_ram:>15.2f} {flash_ram:>15.2f}")
print(f"{'Model (KB)':<25} {'~ (xgboost depends)':>15} {model_kb:>15.1f}")
print("="*100)

# verdict
def verdict_str(a, b, higher_better=True):
    if abs(a - b) < 1e-6:
        return "Tie"
    if higher_better:
        return "awakenFlash" if b > a else "XGBoost"
    else:
        return "awakenFlash" if b < a else "XGBoost"

print("Final winners:")
print("  Accuracy :", verdict_str(xgb_acc, flash_acc, higher_better=True))
print("  Train    :", verdict_str(xgb_train_time, flash_time, higher_better=False))
print("  Inference:", verdict_str(xgb_inf_ms, flash_inf_ms, higher_better=False))
print("="*100)
