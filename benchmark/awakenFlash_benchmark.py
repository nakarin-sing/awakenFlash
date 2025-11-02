#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vs XGBoost — Honest CI Benchmark (No Tweaking)
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import psutil, gc

try:
    from src.awakenFlash_core import train_step, infer, data_stream, N_SAMPLES, N_FEATURES, N_CLASSES, B, H, CONF_THRESHOLD, LS
except ImportError:
    from src.awakenFlash_core import train_step, infer, N_SAMPLES, N_FEATURES, N_CLASSES, B, H, CONF_THRESHOLD, LS
    def data_stream(n=1000):
        X = np.random.randn(n, N_FEATURES).astype(np.float32)
        y = np.random.randint(0, N_CLASSES, size=n).astype(np.int32)
        yield X, y

print("==============================")
print("Starting awakenFlash Benchmark")
print("==============================")

proc = psutil.Process()

# ---------------------------
# Prepare Data
# ---------------------------
X_train = np.random.rand(N_SAMPLES, N_FEATURES)
y_train = np.random.randint(0, N_CLASSES, N_SAMPLES)
X_test = np.random.rand(100_000, N_FEATURES)
y_test = np.random.randint(0, N_CLASSES, 100_000)

# ---------------------------
# XGBoost Baseline
# ---------------------------
t0 = time.time()
xgb = XGBClassifier(n_estimators=200, max_depth=6, n_jobs=-1, random_state=42, tree_method='hist', verbosity=0)
xgb.fit(X_train, y_train)
xgb_train_time = time.time() - t0

t0 = time.time()
xgb_pred = xgb.predict(X_test)
xgb_inf_ms = (time.time() - t0) / len(X_test) * 1000
xgb_acc = accuracy_score(y_test, xgb_pred)

del xgb
gc.collect()

# ---------------------------
# awakenFlash Real Run
# ---------------------------
scale = max(1.0, np.max(np.abs(X_train)) / 127.0)
X_i8 = np.clip(np.round(X_train / scale), -128, 127).astype(np.int8)

mask = np.random.rand(N_FEATURES, H) < 0.7
values = np.random.randint(-4, 5, size=mask.sum()).astype(np.int8)
col_indices = np.where(mask)[1].astype(np.int32)
indptr = np.zeros(H + 1, np.int32)
np.cumsum(np.bincount(np.where(mask)[0], minlength=H), out=indptr[1:])
b1 = np.zeros(H, np.int32)
W2 = np.random.randint(-4, 5, (H, N_CLASSES), np.int8)
b2 = np.zeros(N_CLASSES, np.int32)

t0 = time.time()
values, b1, W2, b2 = train_step(X_i8, y_train, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS)
awaken_train_time = time.time() - t0

X_test_i8 = np.clip(np.round(X_test / scale), -128, 127).astype(np.int8)
t0 = time.time()
pred, ee_ratio = infer(X_test_i8, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD)
awaken_inf_ms = (time.time() - t0) / len(X_test_i8) * 1000
awaken_acc = accuracy_score(y_test, pred)

# ---------------------------
# Results
# ---------------------------
print("\n[XGBoost Results]")
print(f"Accuracy               : {xgb_acc:.4f}")
print(f"Train Time (s)         : {xgb_train_time:.2f}")
print(f"Inference (ms/sample)  : {xgb_inf_ms:.4f}")

print("\n[awakenFlash Results]")
print(f"Accuracy               : {awaken_acc:.4f}")
print(f"Train Time (s)         : {awaken_train_time:.2f}")
print(f"Inference (ms/sample)  : {awaken_inf_ms:.4f}")
print(f"Early Exit Ratio       : {ee_ratio*100:.1f}%")

print("\n==============================")
print("🏁 FINAL VERDICT")
print("==============================")

def verdict(name, awaken_val, xgb_val, higher_is_better=True):
    if awaken_val == xgb_val:
        return "🤝 Tie"
    better = awaken_val > xgb_val if higher_is_better else awaken_val < xgb_val
    symbol = "✅ awakenFlash wins" if better else "❌ XGBoost wins"
    diff = awaken_val - xgb_val if higher_is_better else xgb_val - awaken_val
    return f"{symbol} ({diff:+.4f})"

print(f"Accuracy Result        : {verdict('Accuracy', awaken_acc, xgb_acc)}")
print(f"Train Speed Result     : {verdict('Train', awaken_train_time, xgb_train_time, higher_is_better=False)}")
print(f"Inference Speed Result : {verdict('Inference', awaken_inf_ms, xgb_inf_ms, higher_is_better=False)}")
print("==============================")
