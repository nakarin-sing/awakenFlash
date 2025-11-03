# -*- coding: utf-8 -*-
"""
🌌 STREAMING BENCHMARK — FAIR TEST
---------------------------------
Dataset: Covertype (UCI) ~75 MB
Methods: XGBoost vs Online Ridge (OneStepRLS)
Goal: Stream large dataset without full load
"""

import os
import pandas as pd
import numpy as np
import urllib.request
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

# ==============================
# 1. Download Dataset (Streaming Safe)
# ==============================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
DATA_PATH = "covtype.data.gz"

if not os.path.exists(DATA_PATH):
    print("📥 Downloading dataset...")
    urllib.request.urlretrieve(URL, DATA_PATH)
    print("✅ Download complete")

# ==============================
# 2. Stream Load + Split Chunks
# ==============================
CHUNK_SIZE = 50000
cols = [f"f{i}" for i in range(54)] + ["target"]

results = []

print("🔄 Streaming chunks...")
reader = pd.read_csv(DATA_PATH, compression="gzip", header=None, names=cols, chunksize=CHUNK_SIZE)

# Prepare models
sgd = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, max_iter=1, warm_start=True)
xgb_model = None
chunk_id = 0

for chunk in reader:
    chunk_id += 1
    X = chunk.iloc[:, :-1].values
    y = (chunk.iloc[:, -1] == 2).astype(int)  # binary target

    # --- XGBoost (batch mode)
    dtrain = xgb.DMatrix(X, label=y)
    xgb_params = {"objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0}
    if xgb_model is None:
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=10)
    else:
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=5, xgb_model=xgb_model)

    # --- OneStepRLS (SGD Online)
    sgd.partial_fit(X, y, classes=np.array([0, 1]))

    # --- Evaluation (stream-like)
    y_pred_xgb = (xgb_model.predict(dtrain) > 0.5).astype(int)
    y_pred_sgd = sgd.predict(X)
    auc_xgb = roc_auc_score(y, y_pred_xgb)
    auc_sgd = roc_auc_score(y, y_pred_sgd)

    acc_xgb = accuracy_score(y, y_pred_xgb)
    acc_sgd = accuracy_score(y, y_pred_sgd)

    results.append({
        "chunk": chunk_id,
        "xgb_auc": auc_xgb,
        "sgd_auc": auc_sgd,
        "xgb_acc": acc_xgb,
        "sgd_acc": acc_sgd
    })

    print(f"✅ Chunk {chunk_id:02d}: AUC (XGB={auc_xgb:.3f}, SGD={auc_sgd:.3f})")

# ==============================
# 3. Save results
# ==============================
df_results = pd.DataFrame(results)
df_results.to_csv("results.csv", index=False)
print("\n💾 Results saved to results.csv")
print(df_results.tail())
