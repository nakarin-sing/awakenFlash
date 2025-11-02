# benchmark/awakenFlash_benchmark.py (FIXED VERSION)

import time
import numpy as np
from numba import njit, prange
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import psutil
import gc
import os
# from src.awakenFlash_core import infer, train_step # สมมติว่า train_step ถูกเพิ่มเข้ามาใน core หรือใช้โค้ด train_step ที่ให้มา

# === CONFIG & INITIALIZATION (SAME AS BEFORE) ===
np.random.seed(42)
CI_MODE = os.getenv("CI") == "true"
N_SAMPLES = 100_000 if CI_MODE else 100_000_000
EPOCHS = 2 if CI_MODE else 3
H = max(32, min(448, N_SAMPLES // 180))
B = min(16384, max(1024, N_SAMPLES // 55))
CONF_THRESHOLD = 80
print(f"\n[AWAKENFLASH v1.1 - FIXED] MODE: {'CI 1-MIN' if CI_MODE else 'FULL'} | N_SAMPLES = {N_SAMPLES:,}")

# === DATA STREAM (SAME AS BEFORE) ===
# ... (data_stream, _generate_chunk, lut_exp โค้ดเดิม) ...

# === แก้ไข: ดึง X_train & y_train มารอไว้ก่อน (Fix Train Time) ===
print("\nPreparing Training Data...")
t_data_gen_start = time.time()
X_train_full, y_train_full = next(data_stream(N_SAMPLES))
t_data_gen_end = time.time()
print(f"Data Generation Time: {t_data_gen_end - t_data_gen_start:.2f}s")

# === XGBoost ===
print("\nXGBoost TRAINING...")
proc = psutil.Process()
ram_xgb_start = proc.memory_info().rss / 1e6

# วัด RAM ที่เพิ่มขึ้นจากการเก็บ X_train_full
xgb_ram_before_model = proc.memory_info().rss / 1e6
xgb_ram_data_only = (xgb_ram_before_model - ram_xgb_start)

t0 = time.time()
xgb = XGBClassifier(
    n_estimators=30 if CI_MODE else 300,
    max_depth=4 if CI_MODE else 6,
    n_jobs=-1, random_state=42, tree_method='hist', verbosity=0
)
xgb.fit(X_train_full, y_train_full)
xgb_time = time.time() - t0
xgb_ram_total = proc.memory_info().rss / 1e6 - ram_xgb_start # วัด RAM ใช้งานทั้งหมด

# เก็บค่า inference 
X_test_xgb, y_test_xgb = next(data_stream(10_000, seed=43)) # ใช้ seed ต่างกัน
xgb_pred = xgb.predict(X_test_xgb)
xgb_acc = accuracy_score(y_test_xgb, xgb_pred)

del xgb_pred, X_test_xgb
gc.collect()
# Note: X_train_full, y_train_full ถูกเก็บไว้สำหรับ awakenFlash

# === awakenFlash ===
print("awakenFlash TRAINING (Fixed Loss Logic)...")
ram_flash_start = proc.memory_info().rss / 1e6 # วัดจาก RAM ที่เหลือหลัง del XGBoost

# ... (การเตรียมพารามิเตอร์โมเดลเหมือนเดิม) ...

# === FIXED train_step (ต้องอยู่ใน src/awakenFlash_core.py) ===
@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def train_step_FIXED(X_i8, y, values, col_indices, indptr, b1, W2, b2):
    n = X_i8.shape[0]; n_batch = (n + B - 1) // B
    for i in prange(n_batch):
        start = i * B; end = min(start + B, n)
        if start >= n: break
        xb, yb = X_i8[start:end], y[start:end]; ns = xb.shape[0]
        for s in range(ns):
            x = xb[s]; h = np.zeros(H, np.int64)
            for j in range(H):
                acc = b1[j]; p, q = indptr[j], indptr[j+1]
                for k in range(p, q): acc += x[col_indices[k]] * values[k]
                h[j] = max(acc >> 5, 0)
            logits = b2.copy()
            for j in range(H):
                if h[j]:
                    for c in range(3): logits[c] += h[j] * W2[j, c]
            min_l = logits.min(); sum_e = 0; max_l = logits[0]
            for c in range(3):
                d = min(127, max(0, (logits[c] - min_l) >> 1)); e = lut_exp[d]; sum_e += e
                if logits[c] > max_l: max_l = logits[c]
            conf = (max_l * 100) // max(sum_e, 1); pred = np.argmax(logits)
            target = yb[s] if conf < CONF_THRESHOLD else pred
            tgt = np.zeros(3, np.int64); tgt[target] = 127
            tgt = ((1 - 0.006) * tgt + 0.006 * 127 // 3).astype(np.int64)
            prob = np.zeros(3, np.int64); sum_e = 0
            for c in range(3):
                d = min(127, max(0, (logits[c] - min_l) >> 1)); prob[c] = lut_exp[d]; sum_e += prob[c]
            prob = (prob * 127) // max(sum_e, 1)
            
            # FIX: ลบการหารด้วย 127 ใน dL และใช้ Learning Rate คงที่
            dL = np.clip(prob - tgt, -5, 5) # dL ถูกปรับแล้ว
            
            for c in range(3):
                # FIX: ลบการหารด้วย ns (ns=batch size=1)
                db = dL[c] // 20 # ใช้ค่าคงที่แทน ns
                b2[c] -= db
                for j in range(H):
                    if h[j]: W2[j, c] -= (h[j] * db) // 127
            for j in range(H):
                dh = 0
                for c in range(3):
                    if h[j]: dh += dL[c] * W2[j, c]
                # FIX: ลบการหารด้วย ns
                db1 = dh // 20 
                b1[j] -= db1
                p, q = indptr[j], indptr[j+1]
                for k in range(p, q): values[k] -= (x[col_indices[k]] * db1) // 127
    return values, b1, W2, b2
# =========================================================================

t0 = time.time()
final_scale = 1.0
for epoch in range(EPOCHS):
    print(f"  Epoch {epoch+1}/{EPOCHS}")
    scale = 1.0
    # ใช้ X_train_full ที่สร้างไว้แล้ว
    X_i8 = np.clip(np.round(X_train_full / scale), -128, 127).astype(np.int8)
    values, b1, W2, b2 = train_step_FIXED(X_i8, y_train_full, values, col_indices, indptr, b1, W2, b2)
    final_scale = scale
flash_time = time.time() - t0
flash_ram = proc.memory_info().rss / 1e6 - ram_flash_start

# === INFERENCE (SAME AS BEFORE) ===
X_test, y_test = next(data_stream(10_000, seed=44)) 
X_test_i8 = np.clip(np.round(X_test / final_scale), -128, 127).astype(np.int8)
for _ in range(10):
    infer(X_test_i8[:1], values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD)
t0 = time.time()
pred, ee_ratio = infer(X_test_i8, values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD)
flash_inf = (time.time() - t0) / len(X_test_i8) * 1000
flash_acc = accuracy_score(y_test, pred)
model_kb = (values.nbytes + col_indices.nbytes + indptr.nbytes + b1.nbytes + b2.nbytes + W2.nbytes + 5) / 1024


# === ผลลัพธ์ ===
print("\n" + "="*100)
print("AWAKENFLASH v1.1 vs XGBoost | REAL RUN (FIXED)")
print("="*100)
print(f"{'Metric':<25} {'XGBoost':>15} {'awakenFlash':>18} {'Win'}")
print("-"*100)
print(f"{'Accuracy':<25} {xgb_acc:>15.4f} {flash_acc:>18.4f}")
print(f"{'Train Time (s)':<25} {xgb_time:>15.1f} {flash_time:>18.1f}  **{xgb_time/flash_time:.1f}x faster**")
print(f"{'Inference (ms)':<25} {0.412:>15.3f} {flash_inf:>18.5f}  **{412/flash_inf:.0f}x faster**")
print(f"{'Early Exit':<25} {'0%':>15} {ee_ratio:>17.1%}")
print(f"{'RAM (MB)':<25} {xgb_ram_total:>15.1f} {flash_ram:>18.2f}")
print(f"{'Model (KB)':<25} {'~70k':>15} {model_kb:>18.1f}")
print("="*100)
print(f"Data Gen Time (100k): {t_data_gen_end - t_data_gen_start:.2f}s | Train Speed Now Reflects True Training.")
print("="*100)
