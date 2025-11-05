# awakenFlash Ultimate Benchmark Log Only (Direct Comparison)
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

# ------------------------------
# สมมติคุณมี X_chunks และ y_chunks ของจริงแล้ว
# ------------------------------
# ตัวอย่างสำหรับ testing
num_chunks = 10
n_features = 12
n_samples = 100
X_chunks = [np.random.rand(n_samples, n_features) for _ in range(num_chunks)]
y_chunks = [np.random.randint(0,2,n_samples) for _ in range(num_chunks)]

print("📊 Running Ultimate Temporal Benchmark + XGBoost (Log Only)...\n")

for cid, (X_chunk, y_chunk) in enumerate(zip(X_chunks, y_chunks), start=1):
    # --- Temporal Model ---
    t_start = time.time()
    weights = np.random.rand(X_chunk.shape[1])
    y_pred_temporal = (X_chunk @ weights > weights.sum()/2).astype(int)
    t_time = time.time() - t_start
    t_acc = accuracy_score(y_chunk, y_pred_temporal)
    t_f1 = f1_score(y_chunk, y_pred_temporal)
    t_prec = precision_score(y_chunk, y_pred_temporal)
    t_rec = recall_score(y_chunk, y_pred_temporal)

    # --- XGBoost ---
    x_start = time.time()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_chunk, y_chunk)
    y_pred_xgb = xgb.predict(X_chunk)
    x_time = time.time() - x_start
    x_acc = accuracy_score(y_chunk, y_pred_xgb)
    x_f1 = f1_score(y_chunk, y_pred_xgb)
    x_prec = precision_score(y_chunk, y_pred_xgb)
    x_rec = recall_score(y_chunk, y_pred_xgb)

    # --- Log Detailed Results ---
    print(f"Chunk {cid:02d} Results:")
    print(f"  Temporal -> Acc: {t_acc:.3f} | F1: {t_f1:.3f} | Prec: {t_prec:.3f} | Rec: {t_rec:.3f} | Time: {t_time:.3f}s")
    print(f"  XGBoost  -> Acc: {x_acc:.3f} | F1: {x_f1:.3f} | Prec: {x_prec:.3f} | Rec: {x_rec:.3f} | Time: {x_time:.3f}s")
    print(f"  XGBoost Feature Importances: {xgb.feature_importances_}\n")
