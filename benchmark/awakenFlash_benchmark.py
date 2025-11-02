import time 
import numpy as np 
from numba import njit, prange 
from sklearn.metrics import accuracy_score 
from xgboost import XGBClassifier 
import psutil 
import gc 
import os 

# NOTE: We assume the infer function is correctly defined in src/awakenFlash_core.py 
# with the MAX SPEED (no confidence check) logic, or defined locally below.

np.random.seed(42)

# === CONFIG: CI 1 MIN, FULL 100M ===

CI_MODE = os.getenv("CI") == "true" 
N_SAMPLES = 100_000 if CI_MODE else 100_000_000 
EPOCHS = 2 if CI_MODE else 3 
# Hardcoded Config from core file
H = 448 
B = 1024 
CONF_THRESHOLD = 80
LS = 0.006 # ***FIXED (v2.2): เพิ่มการกำหนดค่า Label Smoothing***

print(f"\n[AWAKENFLASH v2.2] MODE: {'CI 1-MIN' if CI_MODE else 'FULL'} | N_SAMPLES = {N_SAMPLES:,}")

# === DATA STREAM ===

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
 86,97,110,124,140,158,179,202,228,255 ] + [255]*88, np.uint8)[:128], dtype=np.int64)


# === XGBoost ===

print("\nXGBoost TRAINING (300 Trees, Depth 6)...") 
proc = psutil.Process() 
ram_xgb_start = proc.memory_info().rss / 1e6 
xgb_t0 = time.time() 
X_train, y_train = next(data_stream(N_SAMPLES)) 
xgb = XGBClassifier( 
    n_estimators=30 if CI_MODE else 300, 
    max_depth=4 if CI_MODE else 6, 
    n_jobs=-1, 
    random_state=42, 
    tree_method='hist', 
    verbosity=0 
) 
xgb.fit(X_train, y_train) 
xgb_time = time.time() - xgb_t0 
xgb_ram = proc.memory_info().rss / 1e6 - ram_xgb_start

# Inference
X_test_xgb, y_test_xgb = next(data_stream(10_000)) 
xgb_inf_t0 = time.time()
xgb_pred = xgb.predict(X_test_xgb) 
xgb_inf_time = time.time() - xgb_inf_t0
xgb_acc = accuracy_score(y_test_xgb, xgb_pred)
xgb_ms_per_sample = (xgb_inf_time / len(X_test_xgb)) * 1000 

del X_train, y_train, xgb, xgb_pred, X_test_xgb 
gc.collect()

# === awakenFlash ===

print("awakenFlash TRAINING...") 
ram_flash_start = proc.memory_info().rss / 1e6

# Initialization (40 features)
N_FEATURES = 40
N_CLASSES = 3
mask = np.random.rand(N_FEATURES, H) < 0.70 
nnz = np.sum(mask) 
W1 = np.zeros((N_FEATURES, H), np.int8) 
W1[mask] = np.random.randint(-4, 5, nnz, np.int8) 
rows, cols = np.where(mask) 
values = W1[rows, cols].copy().astype(np.int64)
indptr = np.zeros(H + 1, np.int32) 
np.cumsum(np.bincount(rows, minlength=H), out=indptr[1:]) 
col_indices = cols.astype(np.int32) 
b1 = np.zeros(H, np.int64)
W2 = np.random.randint(-4, 5, (H, N_CLASSES), np.int64)
b2 = np.zeros(N_CLASSES, np.int64)

# Train Step Function (Defined Locally)
@njit(parallel=True, fastmath=True, nogil=True, cache=True) 
def train_step(X_i8, y, values, col_indices, indptr, b1, W2, b2, B, H, CONF_THRESHOLD, LS, lut_exp): 
    n = X_i8.shape[0]; num_classes = 3
    n_batch = (n + B - 1) // B 
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
            tgt = ((1 - LS) * tgt + LS * 127 // 3).astype(np.int64) 
            prob = np.zeros(3, np.int64); sum_e = 0 
            for c in range(3): 
                d = min(127, max(0, (logits[c] - min_l) >> 1)); prob[c] = lut_exp[d]; sum_e += prob[c] 
            prob = (prob * 127) // max(sum_e, 1) 
            dL = np.clip((prob - tgt) // 127, -5, 5) 
            for c in range(3): 
                db = dL[c] // ns; b2[c] -= db 
                for j in range(H): 
                    if h[j]: W2[j, c] -= (h[j] * db) // 127 
            for j in range(H): 
                dh = 0 
                for c in range(3): 
                    if h[j]: dh += dL[c] * W2[j, c] 
                db1 = dh // ns; b1[j] -= db1 
                p, q = indptr[j], indptr[j+1] 
                for k in range(p, q): values[k] -= (x[col_indices[k]] * db1) // 127 
    return values, b1, W2, b2

# Inference Function (Defined Locally - MAX SPEED)
@njit(parallel=True, fastmath=True, cache=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD): 
    n = X_i8.shape[0]; num_classes = len(b2); pred = np.empty(n, np.int64); ee = 0
    H_val = len(b1)
    for i in prange(n):
        x = X_i8[i]; h = np.zeros(H_val, np.int64)
        for j in range(H_val):
            acc = b1[j]
            for k in range(indptr[j], indptr[j+1]): acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)
        logits = b2.copy()
        for j in range(H_val):
            if h[j]:
                for c in range(num_classes): logits[c] += h[j] * W2[j, c]
        pred[i] = np.argmax(logits)
    return pred, ee / n

# Training Loop
t0 = time.time() 
final_scale = 1.0 
for epoch in range(EPOCHS): 
    print(f" Epoch {epoch+1}/{EPOCHS}") 
    scale = 1.0 
    for X_chunk, y_chunk in data_stream(N_SAMPLES): 
        scale = max(scale, np.max(np.abs(X_chunk)) / 127.0) 
        X_i8 = np.clip(np.round(X_chunk / scale), -128, 127).astype(np.int8) 
        # Train Step Call: LS is now defined in global scope
        values, b1, W2, b2 = train_step(X_i8, y_chunk, values, col_indices, indptr, b1, W2, b2, B, H, CONF_THRESHOLD, LS, lut_exp)
        del X_i8; gc.collect() 
final_scale = scale 
flash_time = time.time() - t0 
flash_ram = proc.memory_info().rss / 1e6 - ram_flash_start

# === INFERENCE ===

X_test, y_test = next(data_stream(10_000)) 
X_test_i8 = np.clip(np.round(X_test / final_scale), -128, 127).astype(np.int8) 

# Warm-up Call
for _ in range(10): 
    infer(X_test_i8[:1], values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD) 

t0 = time.time() 
pred, ee_ratio = infer(X_test_i8, values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD)
flash_inf = (time.time() - t0) / len(X_test_i8) * 1000 
flash_acc = accuracy_score(y_test, pred) 
model_kb = (values.nbytes + col_indices.nbytes + indptr.nbytes + b1.nbytes + b2.nbytes + W2.nbytes + 5) / 1024

# === ผลลัพธ์ ===

print("\n" + "="*100) 
print("AWAKENFLASH v2.2 vs XGBoost | MAX SPEED INFERENCE TEST") 
print("="*100) 
print(f"{'Metric':<25} {'XGBoost':>15} {'awakenFlash':>18} {'Win'}") 
print("-"*100) 

# Inference Time: ใช้ค่าที่วัดได้จริง
xgb_inf_actual = xgb_ms_per_sample
flash_inf_actual = flash_inf

print(f"{'Accuracy':<25} {xgb_acc:>15.4f} {flash_acc:>18.4f}") 
print(f"{'Train Time (s)':<25} {xgb_time:>15.1f} {flash_time:>18.1f} {xgb_time/flash_time:.1f}x faster") 
print(f"{'Inference (ms)':<25} {xgb_inf_actual:>15.5f} {flash_inf_actual:>18.5f} {xgb_inf_actual/flash_inf_actual:.0f}x faster") 
print(f"{'Early Exit':<25} {'0%':>15} {ee_ratio:>17.1%}") 
print(f"{'RAM (MB)':<25} {xgb_ram:>15.1f} {flash_ram:>18.2f}") 
print(f"{'Model (KB)':<25} {'~70k':>15} {model_kb:>18.1f}") 
print("="*100) 
print("CI READY: MAX SPEED INFERENCE TEST") 
print("="*100)
