#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vs XGBoost — Honest CI Benchmark (v1.7: Error Fix & Speed Re-Validation N_F=40)
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import psutil, gc

# --------------------------------------------------
# Import Core Functions and Configuration
# --------------------------------------------------
try:
    from src.awakenFlash_core import train_step, infer, data_stream, N_SAMPLES, N_FEATURES, N_CLASSES, B, H, CONF_THRESHOLD, LS
except ImportError:
    # **FIXED (v1.7): Fallback definition for data_stream**
    print("Error: Could not import core module. Using dummy config/data.")
    
    # 1. กำหนด CONFIG Fallback (ใช้ค่า N_F=40 ที่ต้องการทดสอบ)
    N_SAMPLES = 100_000
    N_FEATURES = 40
    N_CLASSES = 3
    B = 1024
    H = 448
    CONF_THRESHOLD = 80
    LS = 0.006

    # 2. กำหนด data_stream Fallback (เพื่อแก้ NameError)
    def data_stream(n=1000):
        X = np.random.randn(n, N_FEATURES).astype(np.float32)
        y = np.random.randint(0, N_CLASSES, size=n).astype(np.int32)
        yield X, y
    
    # 3. Dummy Numba Functions (เพื่อไม่ให้เกิด NameError: name 'train_step' is not defined)
    #    หมายเหตุ: ในการรันจริง CI มักจะหา train_step/infer ได้ถ้าไฟล์อยู่ในโครงสร้าง src/
    #    แต่ถ้าหาไม่ได้ ต้องมี definition ตรงนี้ ซึ่งซับซ้อนเกินกว่าจะใส่ไว้ใน except block

# Note: เนื่องจากไม่สามารถนิยาม train_step/infer ภายใน except ได้ง่ายๆ, 
# เราจะถือว่า CI สามารถ import ได้สำเร็จ แต่ถ้าล้มเหลว โค้ดจะ error 
# ในบรรทัดที่เรียกใช้ train_step/infer แทน data_stream
# การแก้ไข data_stream นี้ทำให้ผ่านจุด NameError แรกไปได้

print("==============================")
print("Starting awakenFlash Benchmark (v1.7: SPEED RE-VALIDATION N_F=40)")
print("==============================")

proc = psutil.Process()

# ---------------------------
# Prepare Data (ใช้ N_FEATURES ใหม่)
# ---------------------------
t_data_start = time.time()
# **Error เดิมเกิดขึ้นที่นี่:**
X_train, y_train = next(data_stream(N_SAMPLES)) 
X_test, y_test = next(data_stream(100_000))
t_data_end = time.time()
print(f"Data Generation Time: {t_data_end - t_data_start:.2f}s")


# ---------------------------
# XGBoost Baseline
# ---------------------------
t0 = time.time()
# ใช้ Hyperparameter สำหรับ CI
xgb = XGBClassifier(n_estimators=30, max_depth=4, n_jobs=-1, random_state=42, tree_method='hist', verbosity=0)
xgb.fit(X_train, y_train)
xgb_train_time = time.time() - t0

# Inference Time
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

# Initialization ที่ใช้ N_FEATURES = 40
mask = np.random.rand(N_FEATURES, H) < 0.7
values = np.random.randint(-4, 5, size=mask.sum()).astype(np.int8)
rows, cols = np.where(mask)
col_indices = cols.astype(np.int32)
indptr = np.zeros(H + 1, np.int32)
np.cumsum(np.bincount(rows, minlength=H), out=indptr[1:])
b1 = np.zeros(H, np.int32)
W2 = np.random.randint(-4, 5, (H, N_CLASSES), np.int8)
b2 = np.zeros(N_CLASSES, np.int32)

t0 = time.time()
# Note: ต้องมั่นใจว่า train_step ถูก import มาอย่างถูกต้อง
values, b1, W2, b2 = train_step(X_i8, y_train, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS)
awaken_train_time = time.time() - t0

X_test_i8 = np.clip(np.round(X_test / scale), -128, 127).astype(np.int8)
# Warm-up 
infer(X_test_i8[:1], values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD) 
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
print("🏁 FINAL VERDICT (N_FEATURES=40)")
print("==============================")

def verdict(name, awaken_val, xgb_val, higher_is_better=True):
    if awaken_val == xgb_val:
        return "🤝 Tie"
    better = awaken_val > xgb_val if higher_is_better else awaken_val < xgb_val
    symbol = "✅ awakenFlash wins" if better else "❌ XGBoost wins"
    diff = awaken_val - xgb_val if higher_is_better else xgb_val - awaken_val
    
    if not higher_is_better and awaken_val < xgb_val and xgb_val > 0:
        win_ratio = (xgb_val / awaken_val)
        return f"{symbol} ({win_ratio:.0f}x faster)"
    elif higher_is_better and awaken_val > xgb_val:
        return f"{symbol} (+{diff:.4f})"

    return f"{symbol} ({diff:+.4f})"

print(f"Accuracy Result        : {verdict('Accuracy', awaken_acc, xgb_acc)}")
print(f"Train Speed Result     : {verdict('Train', awaken_train_time, xgb_train_time, higher_is_better=False)}")
print(f"Inference Speed Result : {verdict('Inference', awaken_inf_ms, xgb_inf_ms, higher_is_better=False)}")
print("==============================")
