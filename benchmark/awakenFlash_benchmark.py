#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.12 — UNIFIED FINAL CORE (Skew Correction Test)
"The Ultimate OneStep+ Architecture: Speed, ACC, Stability, and Skew Handling."
MIT © 2025 xAI Research
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import resource
# from scipy.stats import mode # No longer needed

# ========================================
# ONESTEP+ ADAPTIVE CORE (Final Unified Model)
# ========================================

class OneStep:
    """
    Final Unified Adaptive Core 1-Step Model (vΩ.10/vΩ.12)
    Features: Minimal Quadratic Expansion, Adaptive Tikhonov Regularization.
    """
    def _add_minimal_features(self, X):
        X = X.astype(np.float32)
        
        # 1. Base Features (with Bias)
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        
        # 2. Minimal Quadratic Terms (X^2) 
        X_quad = X**2
        
        # 3. Concatenate all features
        return np.hstack([X_b, X_quad])


    def fit(self, X, y):
        X_final = self._add_minimal_features(X)
        y_onehot = np.eye(y.max() + 1, dtype=np.float32)[y]
        
        # Adaptive Tikhonov Regularization (Damping)
        XTX = X_final.T @ X_final
        
        # Adaptive Lambda Calculation
        C = 1e-3 
        lambda_adaptive = C * np.mean(np.diag(XTX)) 
        
        # Solve Linear System: W = (G + lambda*I)^-1 X^T Y
        I = np.eye(XTX.shape[0], dtype=np.float32)
        XTY = X_final.T @ y_onehot
        
        self.W = np.linalg.solve(XTX + lambda_adaptive * I, XTY) 
        
    def predict(self, X):
        X_final = self._add_minimal_features(X)
        return (X_final @ self.W).argmax(axis=1)

# ========================================
# OPTIMIZED BENCHMARK EXECUTION
# ========================================
def benchmark_optimized():
    print(f"RAM Start: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    
    xgb_config = dict(n_estimators=50, max_depth=4, n_jobs=1, verbosity=0, tree_method='hist')
    
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]

    xgb_total_time = 0
    onestep_total_time = 0

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        
        # 💡 NEW: Skew Correction (Log Transform) + Scaling 
        # เพื่อจำลอง Real-World Handling สำหรับข้อมูลที่มีความเบ้ (Skewed)
        
        # 1. Log Transform (Simple Skew Handling - ต้องจัดการค่า 0 ด้วย)
        # ใช้ np.log1p เพื่อจัดการกับค่าที่เป็นศูนย์ (log(1+x))
        X_skew_corrected = np.log1p(X) 
        
        # 2. Standard Scaling (Normalization)
        X_final = (X_skew_corrected - X_skew_corrected.mean(axis=0)) / X_skew_corrected.std(axis=0) 
        
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

        results = []

        # XGBoost (Baseline)
        t0 = time.time()
        model = xgb.XGBClassifier(**xgb_config)
        model.fit(X_train, y_train)
        t_xgb = time.time() - t0
        pred = model.predict(X_test)
        results.append(("XGBoost", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_xgb))
        xgb_total_time += t_xgb

        # OneStep (Final Unified Core)
        t0 = time.time()
        m = OneStep(); m.fit(X_train, y_train)
        t_onestep = time.time() - t0
        pred = m.predict(X_test)
        results.append(("OneStep", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_onestep))
        onestep_total_time += t_onestep

        # PRINT
        print(f"\n===== {name} =====")
        print(f"{'Model':<10} {'ACC':<8} {'F1':<8} {'Time':<8}")
        for r in results:
            print(f"{r[0]:<10} {r[1]:.4f}   {r[2]:.4f}   {r[3]:.4f}s")

    print(f"\nRAM End: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    
    if onestep_total_time > 0:
        speedup = xgb_total_time / onestep_total_time
    else:
        speedup = 0
        
    print("\n" + "="*60)
    print("AWAKEN vΩ.12 — UNIFIED FINAL CORE (Skew Correction Test)")
    print(f"Total Speedup (XGB/OneStep): {speedup:.1f}x")
    print("Goal: Confirm ACC stability after applying Skew Correction.")
    print("============================================================")

if __name__ == "__main__":
    benchmark_optimized()
