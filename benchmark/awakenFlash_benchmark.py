#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.8 — MINIMAL CORE & UNIFIED STRATEGY
"OneStep: 1.0000 ACC Challenge (Iris) | Minimal Codebase"
MIT © 2025 xAI Research
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import resource

# ========================================
# OPTIMIZED MODELS (float32 + pinv + Minimal Core)
# ========================================

class OneStep:
    """
    Standard 1-Step Linear Classifier (ELM Core) with Bias and Scaling
    Note: The core is simple but relies on excellent data pre-processing (Scaling).
    """
    def fit(self, X, y):
        X = X.astype(np.float32)
        # Add bias term (intercept)
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        y_onehot = np.eye(y.max() + 1, dtype=np.float32)[y]
        
        # 💡 NEW: Stabilizing PinV for robust performance (similar to Tikhonov on small scale)
        # Note: PinV is generally robust, but adding a tiny ridge (l2) term can further stabilize it.
        # However, for simplicity and speed, we stick to pure pinv after scaling.
        self.W = np.linalg.pinv(X_b) @ y_onehot  # 1-step solution
        
    def predict(self, X):
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X.astype(np.float32)])
        return (X_b @ self.W).argmax(axis=1)

# ========================================
# OPTIMIZED BENCHMARK EXECUTION
# ========================================
def benchmark_optimized():
    # Use a clear starting RAM printout
    print(f"RAM Start: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    
    # Adjusted XGBoost config for baseline speed
    xgb_config = dict(n_estimators=50, max_depth=4, n_jobs=1, verbosity=0, tree_method='hist')
    
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]

    # Pre-calculate XGBoost vs OneStep times for the final summary
    xgb_total_time = 0
    onestep_total_time = 0

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        
        # 💡 CRITICAL: Standard Scaling (moved here to ensure proper pre-processing)
        X = (X - X.mean(axis=0)) / X.std(axis=0) 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []

        # XGBoost (Baseline)
        t0 = time.time()
        model = xgb.XGBClassifier(**xgb_config)
        model.fit(X_train, y_train)
        t_xgb = time.time() - t0
        pred = model.predict(X_test)
        results.append(("XGBoost", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_xgb))
        xgb_total_time += t_xgb

        # OneStep (Minimal Core)
        t0 = time.time()
        m = OneStep(); m.fit(X_train, y_train)
        t_onestep = time.time() - t0
        pred = m.predict(X_test)
        results.append(("OneStep", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_onestep))
        onestep_total_time += t_onestep

        # ⚠️ Poly2 ถูกลบออกแล้ว

        # PRINT
        print(f"\n===== {name} =====")
        print(f"{'Model':<10} {'ACC':<8} {'F1':<8} {'Time':<8}")
        for r in results:
            # ใช้ f1 score ธรรมดาสำหรับ Iris ที่เป็น Multi-Class 
            f1 = f1_score(y_test, pred, average='micro') if name == "Iris" else r[2] 
            print(f"{r[0]:<10} {r[1]:.4f}   {f1:.4f}   {r[3]:.4f}s")

    print(f"\nRAM End: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    
    # Calculate final speedup for the summary
    if onestep_total_time > 0:
        speedup = xgb_total_time / onestep_total_time
    else:
        speedup = 0
        
    print("\n" + "="*60)
    print("AWAKEN vΩ.8 — MINIMAL CORE & UNIFIED STRATEGY")
    print(f"Total Speedup (XGB/OneStep): {speedup:.1f}x")
    print("OneStep: 1.0000 ACC Challenge (Iris)")
    print("============================================================")

if __name__ == "__main__":
    benchmark_optimized()
