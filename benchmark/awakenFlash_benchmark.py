# -*- coding: utf-8 -*-
"""
Non-Logic Streaming Pipeline (Pseudo Non-Logic AI)
- Context-aware
- Adaptive ensemble
- Graph+Spiking inspired reasoning
- Streaming chunk evaluation
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from xgboost import XGBClassifier
from collections import deque
import random

# ------------------------------
# 1. Create synthetic dataset
# ------------------------------
X, y = make_classification(n_samples=50000, n_features=54, n_informative=40,
                           n_classes=7, random_state=42)
X = StandardScaler().fit_transform(X)

# Split into 10 chunks
chunk_size = 5000
chunks_X = [X[i*chunk_size:(i+1)*chunk_size] for i in range(10)]
chunks_y = [y[i*chunk_size:(i+1)*chunk_size] for i in range(10)]

# ------------------------------
# 2. Non-Logic context memory
# ------------------------------
class NonLogicMemory:
    def __init__(self, max_len=10000):
        self.memory = deque(maxlen=max_len)
    
    def add(self, x, y_pred):
        self.memory.append((x, y_pred))
    
    def retrieve(self, x, top_k=5):
        if not self.memory:
            return []
        # similarity = negative euclidean distance
        sims = [(np.linalg.norm(x - m[0]), m[1]) for m in self.memory]
        sims.sort(key=lambda t: t[0])
        return [pred for _, pred in sims[:top_k]]

context_memory = NonLogicMemory(max_len=10000)

# ------------------------------
# 3. Define Non-Logic predictor
# ------------------------------
def non_logic_predict(x, base_models, memory=context_memory):
    # Base ensemble predictions
    preds = [model.predict(x.reshape(1, -1))[0] for model in base_models]
    
    # Retrieve context-aware predictions
    context_preds = memory.retrieve(x)
    if context_preds:
        # Weighted vote: 70% context, 30% base
        combined = preds + context_preds*3  # amplify context influence
        pred = max(set(combined), key=combined.count)
    else:
        pred = max(set(preds), key=preds.count)
    
    return pred

# ------------------------------
# 4. Initialize base models
# ------------------------------
sgd = SGDClassifier(max_iter=1000, tol=1e-3)
pa = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

base_models = [sgd, pa, xgb]

# ------------------------------
# 5. Streaming training & evaluation
# ------------------------------
results = []

for i, (X_chunk, y_chunk) in enumerate(zip(chunks_X, chunks_y)):
    # Split train/test within chunk
    split = int(0.8 * len(X_chunk))
    X_train, X_test = X_chunk[:split], X_chunk[split:]
    y_train, y_test = y_chunk[:split], y_chunk[split:]
    
    # Train base models (incremental for online)
    for model in base_models:
        try:
            model.partial_fit(X_train, y_train, classes=np.unique(y))
        except AttributeError:
            # xgb doesn't support partial_fit -> fit on current chunk
            model.fit(X_train, y_train)
    
    # Predict with Non-Logic
    y_pred = np.array([non_logic_predict(x, base_models) for x in X_test])
    
    # Update memory
    for x, yp in zip(X_test, y_pred):
        context_memory.add(x, yp)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append((i+1, acc, f1))
    print(f"Chunk {i+1} | Non-Logic Acc={acc:.3f}, F1={f1:.3f}")

# ------------------------------
# 6. Compare with XGBoost alone
# ------------------------------
for i, (X_chunk, y_chunk) in enumerate(zip(chunks_X, chunks_y)):
    split = int(0.8 * len(X_chunk))
    X_test = X_chunk[split:]
    y_test = y_chunk[split:]
    y_pred_xgb = xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')
    print(f"Chunk {i+1} | XGBoost Acc={acc_xgb:.3f}, F1={f1_xgb:.3f}")
