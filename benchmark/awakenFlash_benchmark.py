#!/usr/bin/env python3
# awakenFlash_benchmark_teacher_only.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sunyata_nonlogic import NonLogicLab

# -----------------------------
# Config
# -----------------------------
CHUNK_SIZE = 512
SEED = 2025
N_FEATURES = 20
N_CLASSES = 3
N_SAMPLES = 5000

# -----------------------------
# Synthetic dataset
# -----------------------------
X, y = make_classification(n_samples=N_SAMPLES,
                           n_features=N_FEATURES,
                           n_informative=15,
                           n_redundant=5,
                           n_classes=N_CLASSES,
                           random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Split into chunks
def chunker(X, y, size):
    for i in range(0, X.shape[0], size):
        yield X[i:i+size], y[i:i+size]

# -----------------------------
# Initialize modules
# -----------------------------
lab = NonLogicLab(D_rff=512, ridge=25.0, forget=0.995)
xgb_model = None

# -----------------------------
# Streaming benchmark loop
# -----------------------------
print("=== Streaming Benchmark: NonLogicLab vs XGBoost (Teacher Only) ===")
for chunk_id, (X_chunk, y_chunk) in enumerate(chunker(X_train, y_train, CHUNK_SIZE), 1):
    # --- XGBoost baseline (train incrementally) ---
    if xgb_model is None and X_chunk.shape[0] >= 200:
        dtrain = xgb.DMatrix(X_chunk, label=y_chunk)
        params = {"objective":"multi:softprob", "num_class":N_CLASSES,
                  "max_depth":3, "eta":0.3, "verbosity":0}
        xgb_model = xgb.train(params, dtrain, num_boost_round=6)

    # --- NonLogicLab student (DO NOT USE XGBoost for learning!) ---
    metrics = lab.partial_loop(X_chunk, y_chunk, teacher_blend=0.0)  # teacher_blend=0 => ignore XGBoost

    # --- Evaluation per chunk ---
    y_pred_student = lab.predict(X_chunk)
    acc_student = accuracy_score(y_chunk, y_pred_student)

    if xgb_model is not None:
        y_pred_xgb = np.argmax(xgb_model.predict(xgb.DMatrix(X_chunk)), axis=1)
        acc_xgb = accuracy_score(y_chunk, y_pred_xgb)
    else:
        acc_xgb = None

    print(f"[Chunk {chunk_id}] NonLogicLab acc={acc_student:.4f}, XGBoost acc={acc_xgb}, "
          f"cohesion={metrics['cohesion']:.4f}, action={metrics['action']}")

# -----------------------------
# Final evaluation on test set
# -----------------------------
y_test_pred_student = lab.predict(X_test)
acc_student_final = accuracy_score(y_test, y_test_pred_student)

if xgb_model is not None:
    y_test_pred_xgb = np.argmax(xgb_model.predict(xgb.DMatrix(X_test)), axis=1)
    acc_xgb_final = accuracy_score(y_test, y_test_pred_xgb)
else:
    acc_xgb_final = None

print("\n=== Final Test Accuracy ===")
print(f"NonLogicLab: {acc_student_final:.4f}")
print(f"XGBoost   : {acc_xgb_final}")
