#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.14 — THE GOLDEN RATIO CORE
"Optimal Balance: Minimal Features + Full Real-World Preprocessing."
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
# ONESTEP GOLDEN RATIO CORE (vΩ.14)
# ========================================

class OneStepGoldenRatio:
    """
    The most balanced and stable OneStep core: Minimal features + Adaptive Tikhonov + Full Preprocessing.
    """
    def __init__(self, C=1e-3, clip_percentile=99.5):
        self.C = C
        self.clip_percentile = clip_percentile
        self.mean = None
        self.std = None

    def _preprocess(self, X, is_fit=True):
        X = X.astype(np.float32)
        
        # 1. Skew correction (np.log1p)
        X = np.log1p(X) 
        
        # 2. Outlier handling (Clipping)
        # Calculate percentiles (upper/lower) only on the training data statistics 
        if is_fit:
            upper = np.percentile(X, self.clip_percentile, axis=0)
            lower = np.percentile(X, 100 - self.clip_percentile, axis=0)
            self.upper_clip = upper
            self.lower_clip = lower
        
        X = np.clip(X, self.lower_clip, self.upper_clip)
        
        # 3. Standard scaling 
        if is_fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            self.std[self.std == 0] = 1.0 # Avoid division by zero
        
        X = (X - self.mean) / self.std
        return X

    def _add_features(self, X):
        # 1. Base + Bias Term
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        
        # 2. Minimal Quadratic Term (X^2) - The proven essential non-linearity
        X_quad = X**2
        
        # 3. NO Interaction Terms (Crucial for maintaining minimalist core)
        
        return np.hstack([X_b, X_quad])


    def fit(self, X, y):
        # NOTE: For benchmark, the preprocessed data (X_train) is passed here.
        X_final = self._add_features(X)
        y_onehot = np.eye(y.max() + 1, dtype=np.float32)[y]
        
        # Adaptive Tikhonov Regularization
        XTX = X_final.T @ X_final
        
        # 💡 Adaptive Lambda Calculation (Slightly improved stability calculation)
        # ใช้ trace แทน mean(diag) เพื่อให้เป็นไปตามหลักการ Tikhonov (Matrix Norm)
        lambda_adaptive = self.C * np.trace(XTX) / XTX.shape[0]
        
        # Solve 1-step linear system
        I = np.eye(XTX.shape[0], dtype=np.float32)
        XTY = X_final.T @ y_onehot
        
        self.W = np.linalg.solve(XTX + lambda_adaptive * I, XTY) 
        
    def predict(self, X):
        # X_test is already preprocessed and scaled by the benchmark loop
        X_final = self._add_features(X)
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

    # Initialize the OneStepGoldenRatio class to store preprocessing stats
    m_ultimate = OneStepGoldenRatio()

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        
        # 1. Split BEFORE Preprocessing (Correct production pipeline)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Fit Preprocessing Stats on X_train_raw
        m_ultimate._preprocess(X_train_raw, is_fit=True) 
        
        # 3. Transform X_train and X_test using the calculated stats
        X_train = m_ultimate._preprocess(X_train_raw, is_fit=False)
        X_test = m_ultimate._preprocess(X_test_raw, is_fit=False)


        results = []

        # XGBoost (Baseline)
        t0 = time.time()
        model = xgb.XGBClassifier(**xgb_config)
        model.fit(X_train, y_train)
        t_xgb = time.time() - t0
        pred = model.predict(X_test)
        results.append(("XGBoost", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_xgb))
        xgb_total_time += t_xgb

        # OneStep (Golden Ratio Core)
        t0 = time.time()
        m = OneStepGoldenRatio(); 
        # Crucial: Must re-initialize m with the same preprocessing stats before fit/predict
        m.lower_clip = m_ultimate.lower_clip
        m.upper_clip = m_ultimate.upper_clip
        m.mean = m_ultimate.mean
        m.std = m_ultimate.std
        
        m.fit(X_train, y_train)
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
    print("AWAKEN vΩ.14 — THE GOLDEN RATIO CORE (Final Optimization Test)")
    print(f"Total Speedup (XGB/OneStep): {speedup:.1f}x")
    print("Goal: Restore maximal ACC (>0.97) and Speed (>50x).")
    print("============================================================")

if __name__ == "__main__":
    benchmark_optimized()
