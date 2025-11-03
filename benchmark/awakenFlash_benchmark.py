# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark_streaming.py
Real-World Streaming Benchmark (Covertype)
------------------------------------------
✅ Streaming download direct from UCI (no local file)
✅ Compare: XGBoost, Scikit-learn SGD, OneStepRLS
✅ Save benchmark_results.csv
"""

import pandas as pd
import numpy as np
import time, io, requests, os
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("⚠️ XGBoost not found. Install it for full comparison.")

# =========================
# 1. Streaming Data Loader
# =========================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
CHUNKSIZE = 50000  # about 5 MB per chunk

print(f"🔗 Loading dataset streaming from: {URL}")
t0 = time.time()

reader = pd.read_csv(URL, compression='gzip', header=None, chunksize=CHUNKSIZE)
n_chunks = 0
records = []

# =========================
# 2. Initialize models
# =========================
scaler = StandardScaler()
sgd = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal", warm_start=True)
W = None  # manual OneStepRLS weight

# =========================
# 3. Streaming Training Loop
# =========================
for chunk in reader:
    n_chunks += 1
    X = chunk.iloc[:, :-1].values
    y = chunk.iloc[:, -1].values

    # scale incrementally
    scaler.partial_fit(X)
    X = scaler.transform(X)

    # split small test portion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # ---- scikit-learn SGD (Online) ----
    sgd.partial_fit(X_train, y_train, classes=np.unique(y))
    y_pred = sgd.predict(X_test)
    acc_sgd = accuracy_score(y_test, y_pred)
    f1_sgd = f1_score(y_test, y_pred, average="macro")

    # ---- Simple One-Step RLS ----
    if W is None:
        W = np.zeros(X_train.shape[1])
    pred_rls = X_train @ W
    err = y_train - pred_rls
    gain = np.mean(X_train, axis=0)
    W += 0.0001 * gain * np.mean(err)
    acc_rls = accuracy_score(y_test, (X_test @ W > 0.5).astype(int))
    f1_rls = f1_score(y_test, (X_test @ W > 0.5).astype(int), average="macro")

    records.append([n_chunks, acc_sgd, f1_sgd, acc_rls, f1_rls])
    print(f"Chunk {n_chunks:02d}: SGD acc={acc_sgd:.3f}, RLS acc={acc_rls:.3f}")

    if n_chunks >= 10:
        break  # just for demo (10 chunks ≈ 50 MB)

print(f"✅ Streaming complete in {time.time() - t0:.2f} sec")

# =========================
# 4. Optional XGBoost (batch)
# =========================
if xgb is not None:
    print("🚀 Training XGBoost baseline on 10 chunks...")
    df = pd.concat([chunk for chunk in pd.read_csv(URL, compression='gzip', header=None, nrows=CHUNKSIZE*10)])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=10)
    preds = (model.predict(dtrain) > 0.5).astype(int)
    acc_xgb = accuracy_score(y, preds)
    f1_xgb = f1_score(y, preds, average="macro")
    records.append(["XGB", acc_xgb, f1_xgb, None, None])
    print(f"✅ XGBoost acc={acc_xgb:.3f}, f1={f1_xgb:.3f}")

# =========================
# 5. Save results
# =========================
os.makedirs("benchmark_results", exist_ok=True)
df_results = pd.DataFrame(records, columns=["chunk", "acc_sgd", "f1_sgd", "acc_rls", "f1_rls"])
out_path = "benchmark_results/streaming_results.csv"
df_results.to_csv(out_path, index=False)
print(f"📊 Saved results → {out_path}")
