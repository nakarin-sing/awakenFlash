#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vs XGBoost — Honest CI Benchmark (Speed Re-Validation)
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import psutil, gc

# Import config และ core functions จากไฟล์ src.awakenFlash_core (ที่ถูกแก้ N_FEATURES/N_CLASSES แล้ว)
try:
    from src.awakenFlash_core import train_step, infer, data_stream, N_SAMPLES, N_FEATURES, N_CLASSES, B, H, CONF_THRESHOLD, LS
except ImportError:
    # Fallback กรณีไม่ได้รันในโครงสร้างที่ถูกต้อง
    # (ใน CI จริงจะไม่มีปัญหานี้)
    print("Error: Could not import core module. Using dummy config/data.")
    N_SAMPLES = 100000; N_FEATURES = 40; N_CLASSES = 3; H = 448; CONF_THRESHOLD = 80; LS = 0.006
    # ... (ต้องมีการกำหนด train_step, infer, data_stream, H, B, etc. ในที่นี้หากรันแยก) ...

print("==============================")
print("Starting awakenFlash Benchmark (SPEED RE-VALIDATION N_F=40)")
print("==============================")

proc = psutil.Process()

# ---------------------------
# Prepare Data (ใช้ N_FEATURES ใหม่)
# ---------------------------
# Note: ใช้ data_stream ที่สร้างแบบสุ่มเหมือนเดิม
t_data_start = time.time()
X_train, y_train = next(data_stream(N_SAMPLES))
X_test, y_test = next(data_stream(100_000))
t_data_end = time.time()
print(f"Data Generation Time: {t_data_end - t_data_start:.2f}s")


# ---------------------------
# XGBoost Baseline
# ---------------------------
t0 = time.time()
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

# ใช้ N_FEATURES ที่ถูกต้อง
mask = np.random.rand(N_FEATURES, H) < 0.7
values = np.random.randint(-4, 5, size=mask.sum()).astype(np.int8)
rows, cols = np.where(mask)
col_indices = cols.astype(np.int32)
indptr = np.zeros(H + 1, np.int32)
np.cumsum(np.bincount(rows, minlength=H), out=indptr[1:]) # ใช้ rows ในการคำนวณ indptr (ตามโค้ด v1.5)
b1 = np.zeros(H, np.int32)
W2 = np.random.randint(-4, 5, (H, N_CLASSES), np.int8)
b2 = np.zeros(N_CLASSES, np.int32)

t0 = time.time()
# Note: train_step ในโค้ดตัวอย่างล่าสุดของคุณไม่ได้ใช้ B เป็นตัวแบ่ง batch แต่รันแบบ per-sample (prange n)
values, b1, W2, b2 = train_step(X_i8, y_train, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS)
awaken_train_time = time.time() - t0

X_test_i8 = np.clip(np.round(X_test / scale), -128, 127).astype(np.int8)
# Warm-up (เพื่อให้การวัด Inference แม่นยำ)
infer(X_test_i8[:1], values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD) 
t0 = time.time()
pred, ee_ratio = infer(X_test_i8, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD)
awaken_inf_ms = (time.time() - t0) / len(X_test_i8) * 1000
awaken_acc = accuracy_score(y_test, pred)

# ---------------------------
# Results
# ---------------------------
# ... (ส่วนการแสดงผลลัพธ์เหมือนเดิม) ...
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
    # การแสดงผล Win Ratio ที่ชัดเจนขึ้น
    win_ratio = (xgb_val / awaken_val) if not higher_is_better and awaken_val > 0 else (awaken_val / xgb_val)
    if not higher_is_better and awaken_val < xgb_val:
        return f"{symbol} ({win_ratio:.0f}x faster)"
    return f"{symbol} ({diff:+.4f})"

print(f"Accuracy Result        : {verdict('Accuracy', awaken_acc, xgb_acc)}")
print(f"Train Speed Result     : {verdict('Train', awaken_train_time, xgb_train_time, higher_is_better=False)}")
print(f"Inference Speed Result : {verdict('Inference', awaken_inf_ms, xgb_inf_ms, higher_is_better=False)}")
print("==============================")
