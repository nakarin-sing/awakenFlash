#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.14-R — THE RAM-OPTIMIZED CORE
"Goal: Restore maximal ACC/Speed while reducing RAM by 50% via Float16."
MIT © 2025 xAI Research
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import resource
import gc # Import Garbage Collector for explicit memory management

# ========================================
# ONESTEP GOLDEN RATIO CORE (vΩ.14-R)
# ========================================

class OneStepGoldenRatioRAMOptimized:
    """
    vΩ.14 with strict Float16 usage during matrix calculation to reduce peak RAM.
    """
    def __init__(self, C=1e-3, clip_percentile=99.5):
        self.C = C
        self.clip_percentile = clip_percentile
        self.mean = None
        self.std = None
        self.W = None
        self.upper_clip = None
        self.lower_clip = None

    def _preprocess(self, X, is_fit=True):
        # Using float32 for initial processing to maintain precision
        X = X.astype(np.float32) 
        X = np.log1p(X) 
        
        # Outlier handling
        if is_fit:
            self.upper_clip = np.percentile(X, self.clip_percentile, axis=0)
            self.lower_clip = np.percentile(X, 100 - self.clip_percentile, axis=0)
        
        X = np.clip(X, self.lower_clip, self.upper_clip)
        
        # Standard scaling 
        if is_fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            self.std[self.std == 0] = 1.0 
        
        X = (X - self.mean) / self.std
        return X

    def _add_features(self, X):
        # Uses float32 for feature creation
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        X_quad = X**2
        return np.hstack([X_b, X_quad])

    def fit(self, X, y):
        X_final_32 = self._add_features(X)
        y_onehot_32 = np.eye(y.max() + 1, dtype=np.float32)[y]
        
        # --- RAM OPTIMIZATION: Convert to float16 for the heavy matrix calculation ---
        # This reduces the size of X_final, XTX, and XTY by 50% during the solve phase.
        X_final = X_final_32.astype(np.float16)
        y_onehot = y_onehot_32.astype(np.float16)

        # 1. Compute XTX (50% RAM reduction during this peak)
        XTX = X_final.T @ X_final
        
        # 2. Adaptive Tikhonov Regularization (Calculation remains the same)
        lambda_adaptive = self.C * np.trace(XTX) / XTX.shape[0]
        
        # 3. Solve 1-step linear system (XTY is also calculated with Float16)
        I = np.eye(XTX.shape[0], dtype=np.float16)
        XTY = X_final.T @ y_onehot
        
        # Solve the system (Requires casting back to float32 or float64 for np.linalg.solve stability)
        # We solve in the precision needed by the solver, then store W as float32.
        self.W = np.linalg.solve(XTX.astype(np.float64) + (lambda_adaptive * I).astype(np.float64), 
                                 XTY.astype(np.float64)).astype(np.float32)

        # Explicitly collect garbage to ensure float16 matrices are cleared quickly
        del X_final, y_onehot, XTX, XTY, I 
        gc.collect() 
        
    def predict(self, X):
        X_final = self._add_features(X)
        return (X_final @ self.W).argmax(axis=1)

# ========================================
# OPTIMIZED BENCHMARK EXECUTION (RAM Measurement)
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
    m_ultimate = OneStepGoldenRatioRAMOptimized()

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        
        # 1. Split BEFORE Preprocessing
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

        # OneStep (Golden Ratio RAM-Optimized Core)
        t0 = time.time()
        m = OneStepGoldenRatioRAMOptimized(); 
        # Crucial: Must re-initialize m with the same preprocessing stats before fit/predict
        m.lower_clip = m_ultimate.lower_clip
        m.upper_clip = m_ultimate.upper_clip
        m.mean = m_ultimate.mean
        m.std = m_ultimate.std
        
        m.fit(X_train, y_train)
        t_onestep = time.time() - t0
        pred = m.predict(X_test)
        results.append(("RAM-Opt", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_onestep))
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
        
    print("\n" + "="*70)
    print("AWAKEN vΩ.14-R — THE RAM-OPTIMIZED CORE (Final Test)")
    print(f"Total Speedup (XGB/RAM-Opt): {speedup:.1f}x")
    print("Goal: Check if ACC/Speed maintained while RAM peak is reduced.")
    print("======================================================================")

if __name__ == "__main__":
    benchmark_optimized()
