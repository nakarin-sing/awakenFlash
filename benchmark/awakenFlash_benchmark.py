import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

# ฟังก์ชันคำนวณ metrics
def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro')
    }

# สมมติว่า X_chunks, y_chunks เป็น list ของ dataset ต่อ chunk
# Temporal model function (placeholder)
def run_temporal(X_train, y_train, X_test):
    # ตัวอย่าง weights แบบสุ่ม (12 features)
    weights = np.random.rand(X_train.shape[1])
    # ตัวอย่าง prediction แบบ weighted sum threshold 0.5
    scores = X_test.dot(weights)
    preds = (scores > 0.5).astype(int)
    return preds, weights

print("📊 Running Ultimate Temporal Benchmark + XGBoost (Log Only)...\n")

for cid, (X_chunk, y_chunk) in enumerate(zip(X_chunks, y_chunks), start=1):
    # แยก train/test สมมติ split 80/20
    n_train = int(0.8 * len(X_chunk))
    X_train, X_test = X_chunk[:n_train], X_chunk[n_train:]
    y_train, y_test = y_chunk[:n_train], y_chunk[n_train:]

    # ---- Temporal ----
    start = time.time()
    temporal_pred, temporal_weights = run_temporal(X_train, y_train, X_test)
    temporal_time = time.time() - start
    temporal_metrics = compute_metrics(y_test, temporal_pred)

    print(f"Chunk {cid:02d} Temporal Results:")
    print(f"  Accuracy : {temporal_metrics['accuracy']:.6f}")
    print(f"  F1 Score : {temporal_metrics['f1']:.6f}")
    print(f"  Precision: {temporal_metrics['precision']:.6f}")
    print(f"  Recall   : {temporal_metrics['recall']:.6f}")
    print(f"  Time     : {temporal_time:.3f}s")
    print(f"  Model Weights: {temporal_weights}\n")

    # ---- XGBoost ----
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    start = time.time()
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_time = time.time() - start
    xgb_metrics = compute_metrics(y_test, xgb_pred)

    print(f"Chunk {cid:02d} XGBoost Results:")
    print(f"  Accuracy : {xgb_metrics['accuracy']:.6f}")
    print(f"  F1 Score : {xgb_metrics['f1']:.6f}")
    print(f"  Precision: {xgb_metrics['precision']:.6f}")
    print(f"  Recall   : {xgb_metrics['recall']:.6f}")
    print(f"  Time     : {xgb_time:.3f}s\n")
