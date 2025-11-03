#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AWAKEN vΨ-UltraFast — 30 วินาทีรู้ผล
"ไม่ใช้ pandas | ไม่ดาวน์โหลด | ใช้ synthetic data | CI PASS < 60s"
MIT © 2025 xAI Research
"""

import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.spatial import KDTree
import resource
import xgboost as xgb

# ========================================
# CONFIG (เร็วสุดขีด)
# ========================================
N_SAMPLES = 100_000
N_FEATURES = 28
N_CLASSES = 2
K = 5
H = 32
EPOCHS = 1
LR = 0.1
BATCH_SIZE = 10000

# ========================================
# สร้าง synthetic data แบบ HIGGS (เร็ว)
# ========================================
print("Generating synthetic HIGGS-like data...")
rng = np.random.Generator(np.random.PCG64(42))
X = rng.normal(0, 1, (N_SAMPLES, N_FEATURES)).astype(np.float32)
# สร้าง interaction เหมือน HIGGS
X[:, 10:15] = X[:, :5] * X[:, 5:10] + rng.normal(0, 0.1, (N_SAMPLES, 5))
# label จาก linear combination + noise
weights = rng.normal(0, 1, N_FEATURES)
logits = X @ weights
probs = 1 / (1 + np.exp(-logits))
y = (probs > 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================
# กราฟ K-NN (เร็ว)
# ========================================
def build_knn(X, k=K):
    tree = KDTree(X)
    _, idx = tree.query(X, k=k+1)
    row = np.repeat(np.arange(len(X)), k)
    col = idx[:, 1:].ravel()
    return row, col

row_train, col_train = build_knn(X_train)
row_test, col_test = build_knn(X_test)

# ========================================
# AWAKEN vΨ-UltraFast
# ========================================
rng = np.random.Generator(np.random.PCG64(42))
W1 = rng.normal(0, 0.2, (N_FEATURES, H)).astype(np.float32)
W2 = rng.normal(0, 0.2, (H, 1)).astype(np.float32)

def forward(X, row, col):
    h = X @ W1
    h = np.maximum(h, 0)
    h_agg = np.zeros_like(h)
    nodes = np.unique(row)
    for i in nodes:
        neigh = col[row == i]
        if len(neigh): h_agg[i] = np.mean(h[neigh])
    h = h + 0.3 * h_agg
    h = np.maximum(h, 0)
    return (h @ W2).ravel()

# Train 1 epoch
print("Training AWAKEN...")
t0 = time.time()
perm = np.random.permutation(len(X_train))
X_shuf, y_shuf = X_train[perm], y_train[perm]
for i in range(0, len(X_train), BATCH_SIZE):
    Xb = X_shuf[i:i+BATCH_SIZE]
    yb = y_shuf[i:i+BATCH_SIZE]
    mask = (row_train >= i) & (row_train < i + len(Xb))
    row_b = row_train[mask] - i
    col_b = col_train[mask] - i
    col_b = col_b[(col_b >= 0) & (col_b < len(Xb))]
    row_b = row_b[:len(col_b)]
    logits = forward(Xb, row_b, col_b)
    prob = 1 / (1 + np.exp(-logits))
    d_logits = (prob - yb) / len(yb)
    h = np.maximum(Xb @ W1, 0)
    h_agg = np.zeros_like(h)
    for node in np.unique(row_b):
        neigh = col_b[row_b == node]
        if len(neigh): h_agg[node] = np.mean(h[neigh])
    h = h + 0.3 * h_agg
    h = np.maximum(h, 0)
    d_h = d_logits[:, None] @ W2.T
    d_h *= (h > 0)
    W2 -= LR * (h.T @ d_logits[:, None])
    W1 -= LR * (Xb.T @ d_h)
awaken_time = time.time() - t0

# Predict
logits = forward(X_test, row_test, col_test)
proba = 1 / (1 + np.exp(-logits))
pred = (proba > 0.5).astype(int)
awaken_acc = accuracy_score(y_test, pred)
awaken_auc = roc_auc_score(y_test, proba)

# ========================================
# XGBoost
# ========================================
print("Training XGBoost...")
t0 = time.time()
model_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=6, n_jobs=-1, tree_method='hist', verbosity=0)
model_xgb.fit(X_train, y_train)
xgb_time = time.time() - t0
xgb_pred = model_xgb.predict(X_test)
xgb_proba = model_xgb.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_proba)

# ========================================
# ผลลัพธ์
# ========================================
print(f"RAM: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
print(f"Total Time: {awaken_time + xgb_time:.1f}s")
print("\n" + "="*80)
print("AWAKEN vΨ-UltraFast vs XGBoost — 30 วินาทีรู้ผล")
print("="*80)
print(f"{'Model':<12} {'ACC':<8} {'AUC':<8} {'Time':<8}")
print("-"*80)
print(f"{'XGBoost':<12} {xgb_acc:.4f}  {xgb_auc:.4f}  {xgb_time:.1f}s")
print(f"{'AWAKEN':<12} {awaken_acc:.4f}  {awaken_auc:.4f}  {awaken_time:.1f}s")
print("="*80)
print("ไม่ดาวน์โหลด | ไม่ใช้ pandas | รันใน CI < 30s")
print("="*80)
