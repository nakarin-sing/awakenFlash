#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FULL FAIRNESS BENCHMARK - Complete Version
- Scenario 1: Streaming (Online Learning)
- Scenario 2: Batch Learning (XGBoost vs SGD/PA)
- Scenario 3: Concept Drift
- Scenario 4: Adaptive Streaming Learning
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ✅ FIX: Correct import path for AdaptiveSRLS
from awakenFlash.adaptive_srls import AdaptiveSRLS


def load_data(n_chunks=10):
    """Load and prepare Covertype dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1  # labels 0-6
    
    # Normalize features
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    # Split into chunks
    chunk_size = 10000
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, len(X_all), chunk_size)]
    
    return chunks[:n_chunks], np.unique(y_all)


def scenario1_streaming(chunks, classes):
    print("\n" + "="*70)
    print("🔄 Scenario 1: Streaming (Online Learning)")
    print("="*70)
    
    sgd = SGDClassifier(loss="log_loss", learning_rate="optimal", eta0=0.01, max_iter=5, warm_start=True, random_state=42)
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=5, warm_start=True, random_state=42)
    
    xgb_all_X, xgb_all_y = [], []
    first_sgd = first_pa = True
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        # SGD
        if first_sgd:
            sgd.partial_fit(X_train, y_train, classes=classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train)
        sgd_acc = sgd.score(X_test, y_test)
        
        # PA
        if first_pa:
            pa.partial_fit(X_train, y_train, classes=classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train)
        pa_acc = pa.score(X_test, y_test)
        
        # XGBoost (retrain all chunks)
        xgb_all_X.append(X_train)
        xgb_all_y.append(y_train)
        dtrain = xgb.DMatrix(np.vstack(xgb_all_X), label=np.concatenate(xgb_all_y))
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": len(classes), "max_depth": 4, "eta": 0.3, "verbosity": 0},
            dtrain, num_boost_round=10
        )
        xgb_pred = xgb_model.predict(dtest)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        print(f"Chunk {chunk_id:02d}: SGD={sgd_acc:.3f}, PA={pa_acc:.3f}, XGB={xgb_acc:.3f}")


def scenario2_batch(chunks, classes):
    print("\n" + "="*70)
    print("📦 Scenario 2: Batch Learning")
    print("="*70)
    
    X_all = np.vstack([chunk[0] for chunk in chunks])
    y_all = np.concatenate([chunk[1] for chunk in chunks])
    
    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]
    
    # SGD
    sgd = SGDClassifier(loss="log_loss", max_iter=10, eta0=0.01, learning_rate="constant", random_state=42)
    sgd.fit(X_train, y_train)
    sgd_acc = sgd.score(X_test, y_test)
    
    # PA
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=10, random_state=42)
    pa.fit(X_train, y_train)
    pa_acc = pa.score(X_test, y_test)
    
    # XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(
        {"objective": "multi:softmax", "num_class": len(classes), "max_depth": 6, "eta": 0.1, "verbosity": 0},
        dtrain, num_boost_round=50
    )
    xgb_pred = xgb_model.predict(dtest)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print(f"SGD acc={sgd_acc:.3f}, PA acc={pa_acc:.3f}, XGB acc={xgb_acc:.3f}")


def scenario3_concept_drift(chunks, classes):
    print("\n" + "="*70)
    print("🌊 Scenario 3: Concept Drift")
    print("="*70)
    
    X_train = np.vstack([chunk[0][:8000] for chunk in chunks[:5]])
    y_train = np.concatenate([chunk[1][:8000] for chunk in chunks[:5]])
    X_test = np.vstack([chunk[0][8000:] for chunk in chunks[5:]])
    y_test = np.concatenate([chunk[1][8000:] for chunk in chunks[5:]])
    
    sgd = SGDClassifier(loss="log_loss", max_iter=10, eta0=0.01, learning_rate="constant", random_state=42)
    sgd.fit(X_train, y_train)
    sgd_acc = sgd.score(X_test, y_test)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(
        {"objective": "multi:softmax", "num_class": len(classes), "verbosity": 0},
        dtrain, num_boost_round=30
    )
    xgb_pred = xgb_model.predict(dtest)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print(f"SGD (static) acc={sgd_acc:.3f}, XGB (static) acc={xgb_acc:.3f}")


def scenario4_adaptive():
    print("\n===== Scenario 4: Adaptive Streaming Learning =====")
    
    chunks, classes = load_data(n_chunks=10)
    
    sgd = SGDClassifier(loss="log_loss", eta0=0.01, learning_rate="constant", max_iter=5, warm_start=True, random_state=42)
    a_srls = AdaptiveSRLS()
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        # SGD
        sgd.partial_fit(X_train, y_train, classes=classes)
        sgd_acc = sgd.score(X_test, y_test)
        
        # AdaptiveSRLS
        a_srls.partial_fit(X_train, y_train)
        a_srls_acc = a_srls.score(X_test, y_test)
        
        # XGBoost retrain
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": len(classes), "max_depth": 4, "eta": 0.3, "verbosity": 0},
            dtrain, num_boost_round=10
        )
        xgb_pred = xgb_model.predict(dtest)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        
        print(f"Chunk {chunk_id:02d}: SGD={sgd_acc:.3f}, A-SRLS={a_srls_acc:.3f}, XGB={xgb_acc:.3f}")


def main():
    print("="*70)
    print("🔬 FULL FAIRNESS BENCHMARK")
    print("="*70)
    
    chunks, classes = load_data(n_chunks=10)
    
    scenario1_streaming(chunks, classes)
    scenario2_batch(chunks, classes)
    scenario3_concept_drift(chunks, classes)
    scenario4_adaptive()


if __name__ == "__main__":
    main()
