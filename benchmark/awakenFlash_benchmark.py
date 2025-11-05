# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py
=========================
Real-World Benchmark for awakenFlash
- Tests XGBoost + Scaler + Random Data Simulation
- Saves results into benchmark_results/
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ==============================
# Setup
# ==============================
os.makedirs("benchmark_results", exist_ok=True)

def log(msg: str):
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{t}] {msg}")

# ==============================
# Generate synthetic dataset
# ==============================
def generate_dataset(n_samples=2000, n_features=20, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    y = (np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    return X, y

# ==============================
# Run benchmark
# ==============================
def run_benchmark():
    log("🚀 Starting awakenFlash benchmark...")
    start = time.time()

    X, y = generate_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    split = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]

    # Model setup
    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # Train
    log("Training model...")
    model.fit(X_train, y_train)

    # Predict
    log("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    elapsed = time.time() - start

    # Log results
    log(f"✅ Accuracy: {acc:.4f}")
    log(f"✅ F1 Score: {f1:.4f}")
    log(f"✅ AUC: {auc:.4f}")
    log(f"⏱️ Time Elapsed: {elapsed:.2f}s")

    # Save results
    result = pd.DataFrame([{
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "time_sec": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }])

    result_path = os.path.join("benchmark_results", "awakenFlash_benchmark_results.csv")
    result.to_csv(result_path, index=False)
    log(f"📁 Results saved to: {result_path}")

    log("🏁 Benchmark complete.")
    return result

# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    run_benchmark()
