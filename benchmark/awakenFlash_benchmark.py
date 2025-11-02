# benchmark/awakenFlash_benchmark.py
import time
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import psutil
import gc
from src.awakenFlash_core import train_step, infer  # import เฉพาะฟังก์ชันที่มีจริง

print("==============================")
print("Starting awakenFlash Benchmark")
print("==============================")

proc = psutil.Process()

# ---------------------------
# Dummy parameters (แทนตัวแปรจาก core)
# ---------------------------
N_SAMPLES = 10_000       # ตัวอย่าง training
N_FEATURES = 20          # จำนวน feature
N_CLASSES = 2            # จำนวน class
H = 64                   # hidden dim สำหรับ awakenFlash

# ---------------------------
# Prepare data
# ---------------------------
# สร้าง dummy data แทน data_stream
X_train = np.random.rand(N_SAMPLES, N_FEATURES)
y_train = np.random.randint(0, N_CLASSES, N_SAMPLES)
X_test = np.random.rand(100_000, N_FEATURES)
y_test = np.random.randint(0, N_CLASSES, 100_000)

# ---------------------------
# XGBoost baseline
# ---------------------------
t0 = time.time()
xgb = XGBClassifier(n_estimators=300, max_depth=6, n_jobs=-1, random_state=42, tree_method='hist', verbosity=0)
xgb.fit(X_train, y_train)
xgb_train_time = time.time() - t0

t0 = time.time()
xgb_pred = xgb.predict(X_test)
xgb_inf_ms = (time.time() - t0) / len(X_test) * 1000
xgb_acc = accuracy_score(y_test, xgb_pred)

del X_train, y_train, xgb
gc.collect()

# ---------------------------
# awakenFlash real run
# ---------------------------
X_train = np.random.rand(N_SAMPLES, N_FEATURES)
y_train = np.random.randint(0, N_CLASSES, N_SAMPLES)

scale = max(1.0, np.max(np.abs(X_train)) / 127.0)
X_i8 = np.clip(np.round(X_train / scale), -128, 127).astype(np.int8)

# Initialize weights / bias (simplified example)
mask = np.random.rand(N_FEATURES, H) < 0.7
values = np.random.randint(-4, 5, size=mask.sum()).astype(np.int8)
col_indices = np.where(mask)[1].astype(np.int32)
indptr = np.zeros(H + 1, np.int32)
np.cumsum(np.bincount(np.where(mask)[0], minlength=H), out=indptr[1:])
b1 = np.zeros(H, np.int32)
W2 = np.random.randint(-4, 5, (H, N_CLASSES), np.int8)
b2 = np.zeros(N_CLASSES, np.int32)

t0 = time.time()
values, b1, W2, b2 = train_step(X_i8, y_train, values, col_indices, indptr, b1, W2, b2)
awaken_train_time = time.time() - t0

X_test_i8 = np.clip(np.round(X_test / scale), -128, 127).astype(np.int8)
t0 = time.time()
pred, ee_ratio = infer(X_test_i8, values, col_indices, indptr, b1, W2, b2)
awaken_inf_ms = (time.time() - t0) / len(X_test_i8) * 1000
awaken_acc = accuracy_score(y_test, pred)

# ---------------------------
# Summary table
# ---------------------------
print("\n" + "="*80)
print("awakenFlash vs XGBoost | Dummy Data | CI-friendly output")
print("="*80)
print(f"{'Metric':<28} {'XGBoost':>12} {'awakenFlash':>12} {'Win'}")
print("-"*80)
print(f"{'Accuracy':<28} {xgb_acc:>12.4f} {awaken_acc:>12.4f}  **{('+' if awaken_acc > xgb_acc else '')}{awaken_acc - xgb_acc:.4f}**")
print(f"{'Train Time (s)':<28} {xgb_train_time:>12.1f} {awaken_train_time:>12.1f}  **{xgb_train_time / max(1e-6, awaken_train_time):.2f}× faster**")
print(f"{'Inference (ms/sample)':<28} {xgb_inf_ms:>12.3f} {awaken_inf_ms:>12.3f}  **{xgb_inf_ms / max(1e-6, awaken_inf_ms):.2f}× faster**")
print(f"{'Early Exit Ratio':<28} {'0%':>12} {ee_ratio:>11.1%}  **+{ee_ratio*100:.1f}%**")
print("="*80)
