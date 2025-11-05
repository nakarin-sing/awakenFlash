# ==========================
# awakenFlash_benchmark.py
# Streaming benchmark patch: NonLogicLab vs XGBoost
# ==========================

import time
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sunyata_nonlogic import NonLogicLab

# --------------------------
# Helper: Chunk generator (simulate streaming)
# --------------------------
def generate_chunks(X, y, chunk_size=256):
    n = X.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        yield X[start:end], y[start:end]

# --------------------------
# Benchmark runner
# --------------------------
def run_benchmark(X_train, y_train, X_test, y_test, chunk_size=256):
    # Initialize modules
    lab = NonLogicLab(D_rff=512, ridge=25.0, forget=0.995, teacher_blend=0.35)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # XGBoost baseline
    xgb_params = {"max_depth":3, "eta":0.3, "verbosity":0, "objective":"multi:softprob", "num_class":len(np.unique(y_train))}
    dtrain_full = xgb.DMatrix(X_train, label=y_train)
    xgb_model = xgb.train(xgb_params, dtrain_full, num_boost_round=50)

    print("=== Streaming Benchmark: NonLogicLab vs XGBoost ===")
    chunk_logs = []
    for i, (X_chunk, y_chunk) in enumerate(generate_chunks(X_train, y_train, chunk_size), 1):
        # NonLogicLab partial loop
        metrics = lab.partial_loop(X_chunk, y_chunk)
        y_pred_lab = lab.predict(X_test)
        acc_lab = (y_pred_lab == y_test).mean()

        # XGBoost evaluation
        y_prob_xgb = xgb_model.predict(xgb.DMatrix(X_test))
        y_pred_xgb = np.argmax(y_prob_xgb, axis=1)
        acc_xgb = (y_pred_xgb == y_test).mean()

        log_entry = {
            "chunk": i,
            "lab_acc": acc_lab,
            "lab_metrics": metrics,
            "xgb_acc": acc_xgb
        }
        chunk_logs.append(log_entry)
        print(f"[Chunk {i}] NonLogicLab acc={acc_lab:.4f}, XGBoost acc={acc_xgb:.4f}, Lab action={metrics['action']}")

    print("=== Benchmark Complete ===")
    return chunk_logs

# --------------------------
# Example usage with synthetic data
# --------------------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=5000, n_features=32, n_informative=20, n_classes=4, random_state=42)
    split = int(0.8 * X.shape[0])
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    logs = run_benchmark(X_train, y_train, X_test, y_test, chunk_size=512)
