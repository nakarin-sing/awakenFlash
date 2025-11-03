#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.11.1 — MICRO ENSEMBLE CORE (Mode Fix)
"Fixing the dimension issue in scipy.stats.mode output."
MIT © 2025 xAI Research
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import resource
from scipy.stats import mode # Required for Ensemble Voting

# ========================================
# ONESTEP+ ADAPTIVE CORE (vΩ.10 base)
# ========================================

class OneStep:
    """
    Adaptive Core 1-Step Model with Minimal Quadratic Features and Tikhonov Damping.
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
        
        # Solve Linear System
        I = np.eye(XTX.shape[0], dtype=np.float32)
        XTY = X_final.T @ y_onehot
        
        self.W = np.linalg.solve(XTX + lambda_adaptive * I, XTY) 
        
    def predict(self, X):
        X_final = self._add_minimal_features(X)
        return (X_final @ self.W).argmax(axis=1)

# ========================================
# ONESTEP MICRO ENSEMBLE (vΩ.11.1 Fix)
# ========================================

class OneStepMicroEnsemble:
    """
    Micro Ensemble of N OneStep+ Models trained on bootstrapped samples (Bagging).
    """
    def __init__(self, n_estimators=5, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.estimators = []
        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            # 1. Bootstrapping
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_bag = X[indices]
            y_bag = y[indices]

            # 2. Train OneStep+ Core
            m = OneStep()
            m.fit(X_bag, y_bag)
            self.estimators.append(m)

    def predict(self, X):
        # 3. Prediction and Hard Voting
        predictions = []
        for m in self.estimators:
            predictions.append(m.predict(X))

        predictions = np.array(predictions)
        
        # 💡 FIX: ใช้ .ravel() เพื่อรับประกันว่าผลลัพธ์เป็นอาร์เรย์ 1 มิติเสมอ
        # [0] เพื่อเข้าถึง array ของค่า Mode
        final_preds = mode(predictions, axis=0)[0].ravel() 
        return final_preds

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
    onestep_ensemble_time = 0

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        
        # Pre-processing Strategy: Scaling
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

        # OneStep (Adaptive Core) - เพื่อเปรียบเทียบ
        t0 = time.time()
        m = OneStep(); m.fit(X_train, y_train)
        t_onestep = time.time() - t0
        pred = m.predict(X_test)
        results.append(("OneStep", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_onestep))
        onestep_total_time += t_onestep
        
        # NEW: OneStep Micro Ensemble
        t0 = time.time()
        m_ensemble = OneStepMicroEnsemble(n_estimators=5); m_ensemble.fit(X_train, y_train)
        t_ensemble = time.time() - t0
        pred_ensemble = m_ensemble.predict(X_test)
        
        results.append(("MicroEnsm", accuracy_score(y_test, pred_ensemble), f1_score(y_test, pred_ensemble, average='weighted'), t_ensemble))
        onestep_ensemble_time += t_ensemble

        # PRINT
        print(f"\n===== {name} =====")
        print(f"{'Model':<10} {'ACC':<8} {'F1':<8} {'Time':<8}")
        for r in results:
            print(f"{r[0]:<10} {r[1]:.4f}   {r[2]:.4f}   {r[3]:.4f}s")

    print(f"\nRAM End: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    
    if onestep_ensemble_time > 0:
        speedup = xgb_total_time / onestep_ensemble_time
    else:
        speedup = 0
        
    print("\n" + "="*60)
    print("AWAKEN vΩ.11.1 — MICRO ENSEMBLE CORE (Mode Fix Test)")
    print(f"Total Speedup (XGB/MicroEnsm): {speedup:.1f}x")
    print("Goal: Fix the voting error and verify MicroEnsm performance.")
    print("============================================================")

if __name__ == "__main__":
    benchmark_optimized()
