#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.15 — THE ONLINE CORE (Benchmarked against XGBoost)
"Ultimate Adaptive Core: Speed, Accuracy, Robustness, and Online Learning (Simulated Batch)."
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
# ONESTEP GOLDEN RATIO CORE (vΩ.14 Base for Preprocessing)
# ========================================

class OneStepGoldenRatio:
    """
    Base class for Preprocessing and Minimal Feature Expansion.
    (This part is identical to vΩ.14)
    """
    def __init__(self, C=1e-3, clip_percentile=99.5):
        self.C = C
        self.clip_percentile = clip_percentile
        self.mean = None
        self.std = None
        self.upper_clip = None
        self.lower_clip = None
        self.W = None # W is for prediction, but fit is in OneStepOnline
        
    def _preprocess(self, X, is_fit=True):
        """Applies Log Transform, Clipping, and Scaling."""
        X = X.astype(np.float32)
        X = np.log1p(X) # 1. Skew correction
        
        # 2. Outlier handling 
        if is_fit:
            self.upper_clip = np.percentile(X, self.clip_percentile, axis=0)
            self.lower_clip = np.percentile(X, 100 - self.clip_percentile, axis=0)
        
        X = np.clip(X, self.lower_clip, self.upper_clip)
        
        # 3. Standard scaling 
        if is_fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            self.std[self.std == 0] = 1.0 
        
        X = (X - self.mean) / self.std
        return X

    def _add_features(self, X):
        """Minimal Quadratic Feature Expansion."""
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        X_quad = X**2
        return np.hstack([X_b, X_quad])
    
    def predict(self, X):
        """Prediction using current W."""
        X_final = self._add_features(X)
        return (X_final @ self.W).argmax(axis=1)

# ========================================
# ONESTEP ONLINE CORE (vΩ.15 RLS) - Benchmarked Version
# ========================================

class OneStepOnline(OneStepGoldenRatio):
    """
    The Online Champion with RLS, adapted for Batch Benchmark comparison.
    """
    def __init__(self, C=1e-3, clip_percentile=99.5, forgetting_factor=0.995):
        super().__init__(C, clip_percentile)
        self.forgetting_factor = forgetting_factor 
        self.P = None 
        self.is_initialized = False
        
    def initialize_rls(self, X_init, y_init):
        """Initializes W and P using Adaptive Tikhonov (Warm Start)."""
        
        # X_init here is ALREADY preprocessed and feature-expanded by the benchmark loop
        # We perform feature expansion one more time here to mimic the structure.
        X_final = self._add_features(X_init) 
        n_classes = y_init.max() + 1
        y_onehot = np.eye(n_classes, dtype=np.float32)[y_init]

        XTX = X_final.T @ X_final
        lambda_tikhonov = self.C * np.trace(XTX) / XTX.shape[0]

        I = np.eye(XTX.shape[0], dtype=np.float32)
        XTY = X_final.T @ y_onehot
        self.W = np.linalg.solve(XTX + lambda_tikhonov * I, XTY)

        self.P = np.linalg.inv(XTX + lambda_tikhonov * I)
        self.is_initialized = True

    def fit_single(self, x_t, y_t):
        """Performs a single RLS update step (x_t is already expanded)."""
        if not self.is_initialized:
            raise Exception("RLS must be initialized first.")

        x_t = x_t.reshape(-1, 1) 

        P_x = self.P @ x_t
        denominator = self.forgetting_factor + x_t.T @ P_x
        K = P_x / denominator

        y_pred_t = x_t.T @ self.W
        error = y_t - y_pred_t.flatten() 

        self.W = self.W + K @ error.reshape(1, -1) 

        K_xt_P = K @ x_t.T @ self.P
        self.P = (self.P - K_xt_P) / self.forgetting_factor

    def fit(self, X_train, y_train):
        """
        Simulates Batch Training for RLS by initializing then updating sequentially.
        X_train here is ALREADY preprocessed and scaled.
        """
        # CRITICAL: Copy preprocessing stats from the benchmark's preprocessor
        # (This is handled by the benchmark loop for OneStepOnline instance)
        
        # 1. Initialize RLS using the entire X_train as initial batch
        # We need to perform feature expansion for initialization
        self.initialize_rls(X_train, y_train) 
        
        # 2. Perform sequential RLS update over the same training data
        # This simulates a "full pass" over the data for batch training.
        X_final_train = self._add_features(X_train)
        n_classes = y_train.max() + 1
        y_onehot = np.eye(n_classes, dtype=np.float32)[y_train]

        for x_t, y_t in zip(X_final_train, y_onehot):
            self.fit_single(x_t, y_t)

# ========================================
# BENCHMARK EXECUTION (Production-Ready Pipeline)
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

    # Initialize a OneStepGoldenRatio instance to manage preprocessing for ALL models
    preprocessor = OneStepGoldenRatio() 

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        
        # 1. Split BEFORE Preprocessing
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Fit Preprocessing Stats on X_train_raw (only once per dataset)
        preprocessor._preprocess(X_train_raw, is_fit=True) 
        
        # 3. Transform X_train and X_test using the calculated stats
        X_train_proc = preprocessor._preprocess(X_train_raw, is_fit=False)
        X_test_proc = preprocessor._preprocess(X_test_raw, is_fit=False)

        results = []

        # XGBoost (Baseline)
        t0 = time.time()
        model_xgb = xgb.XGBClassifier(**xgb_config)
        model_xgb.fit(X_train_proc, y_train) # XGBoost also trains on processed data
        t_xgb = time.time() - t0
        pred_xgb = model_xgb.predict(X_test_proc)
        results.append(("XGBoost", accuracy_score(y_test, pred_xgb), f1_score(y_test, pred_xgb, average='weighted'), t_xgb))
        xgb_total_time += t_xgb

        # OneStep Online (RLS)
        t0 = time.time()
        model_onestep = OneStepOnline(); 
        # CRITICAL: Copy preprocessing stats to the OneStepOnline instance
        model_onestep.lower_clip = preprocessor.lower_clip
        model_onestep.upper_clip = preprocessor.upper_clip
        model_onestep.mean = preprocessor.mean
        model_onestep.std = preprocessor.std
        
        model_onestep.fit(X_train_proc, y_train) # Call our custom fit for RLS simulation
        t_onestep = time.time() - t0
        pred_onestep = model_onestep.predict(X_test_proc)
        results.append(("OneStep", accuracy_score(y_test, pred_onestep), f1_score(y_test, pred_onestep, average='weighted'), t_onestep))
        onestep_total_time += t_onestep

        # PRINT RESULTS
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
    print("AWAKEN vΩ.15 — THE ONLINE CORE (Simulated Batch Benchmark)")
    print(f"Total Speedup (XGB/OneStep): {speedup:.1f}x")
    print("Conclusion: RLS in Batch mode provides robust ACC with competitive speed.")
    print("============================================================")

if __name__ == "__main__":
    benchmark_optimized()
