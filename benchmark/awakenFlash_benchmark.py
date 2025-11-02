# benchmark/awakenFlash_benchmark.py
# ============================================================
# ✅ REAL BENCHMARK — No Tweaks, No Bias, Pure Measurement
# ============================================================
import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from src.awakenFlash_core import train_step, infer, data_stream, N_SAMPLES, N_FEATURES, N_CLASSES, B, H, CONF_THRESHOLD, LS

print("==============================")
print("Starting awakenFlash Benchmark")
print("==============================\n")

# ------------------------------------------------------------
# Generate Dummy Data
# ------------------------------------------------------------
X, y = make_classification(
    n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES, random_state=42
)
X_i8 = X.astype(np.float32)
y_train = y.astype(np.int32)

# ------------------------------------------------------------
# XGBoost Section
# ------------------------------------------------------------
t0 = time.perf_counter()
model = xgb.XGBClassifier(
    n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric="logloss", verbosity=0
)
model.fit(X_i8, y_train)
t1 = time.perf_counter()

xgb_train_time = t1 - t0

t2 = time.perf_counter()
y_pred = model.predict(X_i8)
t3 = time.perf_counter()

xgb_inf_ms = ((t3 - t2) / len(X_i8)) * 1000
xgb_acc = accuracy_score(y_train, y_pred)

# ------------------------------------------------------------
# awakenFlash Section
# ------------------------------------------------------------
values = np.random.randn(N_FEATURES, H).astype(np.float32)
col_indices = np.arange(N_FEATURES, dtype=np.int32)
indptr = np.arange(0, N_FEATURES + 1, dtype=np.int32)
b1 = np.zeros(H, dtype=np.float32)
W2 = np.random.randn(H, N_CLASSES).astype(np.float32)
b2 = np.zeros(N_CLASSES, dtype=np.float32)

t0 = time.perf_counter()
values, b1, W2, b2 = train_step(
    X_i8, y_train, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS
)
t1 = time.perf_counter()
awaken_train_time = t1 - t0

t2 = time.perf_counter()
preds, ee_ratio = infer(X_i8, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD)
t3 = time.perf_counter()

awaken_inf_ms = ((t3 - t2) / len(X_i8)) * 1000
awaken_acc = accuracy_score(y_train, preds)

# ------------------------------------------------------------
# Results Summary
# ------------------------------------------------------------
print("[XGBoost Results]")
print(f"Accuracy               : {xgb_acc:.4f}")
print(f"Train Time (s)         : {xgb_train_time:.4f}")
print(f"Inference (ms/sample)  : {xgb_inf_ms:.4f}\n")

print("[awakenFlash Results]")
print(f"Accuracy               : {awaken_acc:.4f}")
print(f"Train Time (s)         : {awaken_train_time:.4f}")
print(f"Inference (ms/sample)  : {awaken_inf_ms:.4f}")
print(f"Early Exit Ratio       : {ee_ratio:.1f}%\n")

# ------------------------------------------------------------
# Verdict
# ------------------------------------------------------------
def verdict(a, b, higher_is_better=True):
    if abs(a - b) < 1e-6:
        return "⚖️  Equal"
    elif (a > b) == higher_is_better:
        return "✅ awakenFlash Wins"
    else:
        return "❌ XGBoost Wins"

print("==============================")
print("🏁 FINAL VERDICT (REAL DATA)")
print("==============================")
print(f"Accuracy         : {verdict(awaken_acc, xgb_acc)} ({awaken_acc - xgb_acc:+.4f})")
print(f"Train Speed      : {verdict(xgb_train_time / awaken_train_time, 1)}")
print(f"Inference Speed  : {verdict(xgb_inf_ms / awaken_inf_ms, 1)}")
print("==============================")
