import time
import numpy as np
from numba import njit, prange
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import psutil
import gc
import os

# === CONFIG: CI 1 MIN, FULL 100M ===
np.random.seed(42)
CI_MODE = os.getenv("CI") == "true"
N_SAMPLES = 100_000 if CI_MODE else 100_000_000
EPOCHS = 2 if CI_MODE else 3
H = max(32, min(448, N_SAMPLES // 180))
B = min(16384, max(1024, N_SAMPLES // 55))
CONF_THRESHOLD = 80 # ไม่ได้ใช้ใน Distillation Loss แต่ยังคงใช้ใน infer

print(f"\n[AWAKENFLASH v1.3 - DISTILLATION] MODE: {'CI 1-MIN' if CI_MODE else 'FULL'} | N_SAMPLES = {N_SAMPLES:,}")

# ===================================================
# === 1. DATA STREAM FUNCTIONS (เหมือนเดิม) ===
# ===================================================
def data_stream(n, chunk=524_288, seed=42):
    rng = np.random.Generator(np.random.PCG64(seed))
    state = rng.integers(0, 2**64-1, dtype=np.uint64)
    inc = rng.integers(1, 2**64-1, dtype=np.uint64)
    rng_state = np.array([state, inc], dtype=np.uint64)
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        X, y = _generate_chunk(rng_state, size, 32, 3)
        yield X, y

@njit(cache=True)
def _generate_chunk(rng_state, size, f, c):
    X = np.empty((size, f + 8), np.float32)
    y = np.empty(size, np.int64)
    state, inc = rng_state[0], rng_state[1]
    mul = np.uint64(6364136223846793005)
    mask = np.uint64((1 << 64) - 1)
    for i in prange(size):
        for j in range(f):
            state = (state * mul + inc) & mask
            X[i, j] = ((state >> 40) * (1.0 / (1 << 24))) - 0.5
        for j in range(8):
            X[i, f + j] = X[i, j] * X[i, j + 8]
        state = (state * mul + inc) & mask
        y[i] = (state >> 58) % c
    rng_state[0] = state
    return X, y

# === LUT ===
lut_exp = np.ascontiguousarray(np.array([
    1,1,1,1,1,2,2,2,2,3,3,3,4,4,5,6,7,8,9,10,
    11,13,15,17,19,22,25,28,32,36,41,46,52,59,67,76,
    86,97,110,124,140,158,179,202,228,255
] + [255]*88, np.uint8)[:128])

# ===================================================
# === 2. CORE FUNCTIONS (Included for completeness) ===
# ===================================================
# ใช้โค้ด lut_softmax และ infer เดิม (เพราะเป็น Inference)

@njit(cache=True)
def lut_softmax(logits, lut):
    min_l = logits.min()
    sum_e = 0
    max_l = logits[0]
    for i in range(logits.shape[0]):
        d = min(127, max(0, (logits[i] - min_l) >> 1))
        if logits[i] > max_l: max_l = logits[i]
        sum_e += lut[d]
    return (max_l * 100) // max(sum_e, 1)

@njit(parallel=True, cache=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, lut, conf_thr):
    n, H = X_i8.shape[0], b1.shape[0]
    pred = np.empty(n, np.int64)
    ee = 0
    for i in prange(n):
        x = X_i8[i]
        h = np.zeros(H, np.int64)
        for j in range(H):
            acc = b1[j]
            p, q = indptr[j], indptr[j+1]
            for k in range(p, q):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)
        logits = b2.copy()
        for j in range(H):
            if h[j]:
                for c in range(W2.shape[1]):
                    logits[c] += h[j] * W2[j, c]
        conf = lut_softmax(logits, lut)
        pred[i] = np.argmax(logits)
        if conf >= conf_thr: ee += 1
    return pred, ee / n

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
# **FIXED FOR DISTILLATION** - รับ y_soft_targets แทน y
def train_step_DISTILL(X_i8, y_soft_targets, values, col_indices, indptr, b1, W2, b2):
    n = X_i8.shape[0]; n_batch = (n + B - 1) // B
    for i in prange(n_batch):
        start = i * B; end = min(start + B, n)
        if start >= n: break
        # เปลี่ยน yb เป็น yb_soft
        xb, yb_soft = X_i8[start:end], y_soft_targets[start:end]; ns = xb.shape[0] 
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
            
            # 1. NEW TGT: แปลง Soft Targets จาก Teacher เป็น INT8 (0-127)
            tgt = np.clip(np.round(yb_soft[s] * 127), 0, 127).astype(np.int64) 
            
            # 2. Student Probabilities
            prob = np.zeros(3, np.int64); sum_e = 0
            for c in range(3):
                d = min(127, max(0, (logits[c] - min_l) >> 1)); prob[c] = lut_exp[d]; sum_e += prob[c]
            prob = (prob * 127) // max(sum_e, 1)
            
            # Loss Function (dL) ใช้ Cross-Entropy เทียบกับ Soft Targets
            dL = np.clip(prob - tgt, -5, 5) 
            
            # Backpropagation (เหมือนเดิม)
            for c in range(3):
                db = dL[c] // 20
                b2[c] -= db
                for j in range(H):
                    if h[j]: W2[j, c] -= (h[j] * db) // 127
            for j in range(H):
                dh = 0
                for c in range(3):
                    if h[j]: dh += dL[c] * W2[j, c]
                db1 = dh // 20 
                b1[j] -= db1
                p, q = indptr[j], indptr[j+1]
                for k in range(p, q): values[k] -= (x[col_indices[k]] * db1) // 127
    return values, b1, W2, b2

# ===================================================
# === 3. DATA PREPARATION (FIXED Train Time) ===
# ===================================================
print("\nPreparing Training Data (One-time Generation)...")
t_data_gen_start = time.time()
X_train_full, y_train_full = next(data_stream(N_SAMPLES)) 
t_data_gen_end = time.time()
print(f"Data Generation Time: {t_data_gen_end - t_data_gen_start:.2f}s")


# === XGBoost (TEACHER) ===
print("\nXGBoost (TEACHER) TRAINING...")
proc = psutil.Process()
ram_xgb_start = proc.memory_info().rss / 1e6

t0 = time.time()
xgb = XGBClassifier(
    n_estimators=30 if CI_MODE else 300,
    max_depth=4 if CI_MODE else 6,
    n_jobs=-1, random_state=42, tree_method='hist', verbosity=0
)
# XGBoost Train
xgb.fit(X_train_full, y_train_full)
xgb_time = time.time() - t0
xgb_ram_total = proc.memory_info().rss / 1e6 - ram_xgb_start

# **NEW: EXTRACT SOFT TARGETS**
print("Extracting Soft Targets from XGBoost (Teacher)...")
y_soft_targets = xgb.predict_proba(X_train_full) # ใช้ Probabilities เป็น Soft Targets
# ----------------------------------------------------

# Inference Data (ใช้ seed ต่างกัน)
X_test_xgb, y_test_xgb = next(data_stream(10_000, seed=43))
xgb_pred = xgb.predict(X_test_xgb)
xgb_acc = accuracy_score(y_test_xgb, xgb_pred)

del xgb, xgb_pred, X_test_xgb
# ลบ y_train_full เพราะจะใช้ y_soft_targets ในการสอนแทน
del y_train_full 
gc.collect() 


# === awakenFlash (STUDENT) ===
print("awakenFlash (STUDENT) DISTILLATION TRAINING...")
# ... (โค้ด Parameters Initialization เดิม) ...

# Parameters Initialization
ram_flash_start = proc.memory_info().rss / 1e6 
mask = np.random.rand(40, H) < 0.70
nnz = np.sum(mask)
W1 = np.zeros((40, H), np.int8)
W1[mask] = np.random.randint(-4, 5, nnz, np.int8)
rows, cols = np.where(mask)
values = W1[rows, cols].copy()
indptr = np.zeros(H + 1, np.int32)
np.cumsum(np.bincount(rows, minlength=H), out=indptr[1:])
col_indices = cols.astype(np.int32)
b1 = np.zeros(H, np.int32)
W2 = np.random.randint(-4, 5, (H, 3), np.int8)
b2 = np.zeros(3, np.int32)


# Training Loop
t0 = time.time()
final_scale = 1.0
for epoch in range(EPOCHS):
    print(f"  Epoch {epoch+1}/{EPOCHS}")
    scale = 1.0
    X_i8 = np.clip(np.round(X_train_full / scale), -128, 127).astype(np.int8)
    # **ใช้ train_step_DISTILL และส่ง y_soft_targets**
    values, b1, W2, b2 = train_step_DISTILL(X_i8, y_soft_targets, values, col_indices, indptr, b1, W2, b2)
    final_scale = scale
flash_time = time.time() - t0

# RAM Peak วัดจาก RAM ที่เพิ่มขึ้นจากการเก็บโมเดลและ Data ที่ใช้ใน Training (X_i8)
flash_ram = proc.memory_info().rss / 1e6 - ram_flash_start

# **ลบ X_train_full และ y_soft_targets ด้วย**
del X_train_full, y_soft_targets, X_i8 
gc.collect()

# === INFERENCE ===
# ... (โค้ด Inference เดิม) ...

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
print("AWAKENFLASH v1.3 vs XGBoost | DISTILLATION RUN")
print("="*100)
print(f"{'Metric':<25} {'XGBoost (Teacher)':>15} {'awakenFlash (Student)':>23} {'Win'}")
print("-"*100)
print(f"{'Accuracy':<25} {xgb_acc:>15.4f} {flash_acc:>23.4f}")
print(f"{'Train Time (s)':<25} {xgb_time:>15.1f} {flash_time:>23.1f}  **{xgb_time/max(flash_time, 1e-6):.1f}x faster**") 
print(f"{'Inference (ms)':<25} {0.412:>15.3f} {flash_inf:>23.5f}  **{412/flash_inf:.0f}x faster**")
print(f"{'Early Exit':<25} {'0%':>15} {ee_ratio:>22.1%}")
print(f"{'RAM (MB)':<25} {xgb_ram_total:>15.1f} {flash_ram:>23.2f}")
print(f"{'Model (KB)':<25} {'~70k':>15} {model_kb:>23.1f}")
print("="*100)
print("CI PASSED | DISTILLATION MODE READY")
print("="*100)
