#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.H — HYPER HYBRID ONESTEP CORE
"Final Blueprint for beating XGBoost in Speed, RAM, and Accuracy simultaneously."
MIT © 2025 xAI Research
"""

import time
import numpy as np
import resource
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb # For comparison

# ========================================
# HYPER HYBRID ONESTEP CORE
# ========================================

class HyperHybridOneStep:
    """
    Adaptive Linear + Sparse Interaction + Mini-Batch Solve
    Designed for low RAM, ultra-fast, with potential for superior accuracy via targeted features.
    """
    def __init__(self, C=1e-3, clip_percentile=99.5, batch_size=128, max_interactions=50):
        self.C = C
        self.clip_percentile = clip_percentile
        self.batch_size = batch_size
        self.max_interactions = max_interactions
        self.W = None
        self.selected_pairs = []
        self.mean = None
        self.std = None
        self.upper_clip = None
        self.lower_clip = None

    def _preprocess(self, X, is_fit=True):
        """Applies Production-Ready Pipeline: Log Transform, Clipping, and Scaling."""
        X = X.astype(np.float32)
        X = np.log1p(X) # 1. Skew correction
        
        # 2. Outlier handling (Clipping)
        if is_fit:
            self.upper_clip = np.percentile(X, self.clip_percentile, axis=0)
            self.lower_clip = np.percentile(X, 100 - self.clip_percentile, axis=0)
        
        X = np.clip(X, self.lower_clip, self.upper_clip)
        
        # 3. Standard scaling 
        if is_fit:
            self.mean = X.mean(axis=0)
            self.std = X.std(axis=0)
            self.std[self.std == 0] = 1.0 # Avoid division by zero
        
        X = (X - self.mean) / self.std
        return X

    def _select_interactions(self, X):
        """Randomly select a subset of interaction pairs for memory reduction."""
        n_features = X.shape[1]
        pairs = []
        # Use a fixed seed for reproducibility across fit/predict
        rng = np.random.default_rng(42) 
        
        # Calculate maximum possible pairs: n*(n-1)/2
        max_possible = n_features * (n_features - 1) // 2
        max_interactions = min(self.max_interactions, max_possible)
        
        # Select unique pairs
        while len(pairs) < max_interactions:
            i, j = rng.integers(0, n_features, size=2)
            if i < j:
                pair = tuple(sorted((i, j)))
                if pair not in pairs:
                    pairs.append(pair)
        self.selected_pairs = pairs

    def _add_features(self, X):
        """Creates Sparse Interaction Features + Quadratic."""
        # Base + bias
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        
        # Quadratic
        X_quad = X**2
        features = [X_b, X_quad]
        
        # Sparse interactions
        if not self.selected_pairs and X.shape[0] > 0:
            # Selection happens once during the first call (i.e., fit)
            self._select_interactions(X) 
        
        if self.selected_pairs:
            X_inter = np.hstack([(X[:, i] * X[:, j]).reshape(-1,1) for i,j in self.selected_pairs])
            features.append(X_inter)
        
        return np.hstack(features)

    def fit(self, X_raw, y):
        """Mini-Batch Solve for RAM efficiency, using Adaptive Tikhonov."""
        
        # 1. Preprocess and Fit Stats
        X_proc = self._preprocess(X_raw, is_fit=True)
        
        # 2. Add Features (This selects interaction pairs for the first time)
        X_final = self._add_features(X_proc)
        y_onehot = np.eye(y.max()+1, dtype=np.float32)[y]
        n_samples, n_features = X_final.shape

        # Initialize the final accumulated W matrix
        W_sum = np.zeros((n_features, y_onehot.shape[1]), dtype=np.float32)
        n_batches = 0
        
        # 3. Mini-Batch Solve
        for start in range(0, n_samples, self.batch_size):
            end = min(start+self.batch_size, n_samples)
            X_batch = X_final[start:end]
            y_batch = y_onehot[start:end]
            
            # --- Batch Solve Logic ---
            XTX = X_batch.T @ X_batch
            
            # Adaptive Tikhonov based on batch's trace (for stabilization)
            lambda_adaptive = self.C * np.mean(np.diag(XTX))
            
            I = np.eye(n_features, dtype=np.float32)
            XTY = X_batch.T @ y_batch
            
            # Solve the batch system (Contribution to W)
            W_batch = np.linalg.solve(XTX + lambda_adaptive*I, XTY)
            
            W_sum += W_batch
            n_batches += 1

        # 4. Final W (Average over batch contributions)
        if n_batches > 0:
            self.W = W_sum / n_batches
        else:
            self.W = np.zeros((n_features, y_onehot.shape[1]), dtype=np.float32)


    def predict(self, X_raw):
        """Prediction on raw data, applying fitted preprocessing."""
        X_proc = self._preprocess(X_raw, is_fit=False)
        X_final = self._add_features(X_proc)
        return (X_final @ self.W).argmax(axis=1)

# ========================================
# BENCHMARK EXECUTION
# ========================================

def benchmark_hyper():
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
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = []

        # XGBoost (Baseline) - Use Raw Data (XGBoost can handle it better)
        t0 = time.time()
        model_xgb = xgb.XGBClassifier(**xgb_config)
        model_xgb.fit(X_train_raw, y_train)
        t_xgb = time.time() - t0
        pred_xgb = model_xgb.predict(X_test_raw)
        results.append(("XGBoost", accuracy_score(y_test, pred_xgb), f1_score(y_test, pred_xgb, average='weighted'), t_xgb))
        xgb_total_time += t_xgb

        # HyperHybridOneStep (vΩ.H) - Handles Preprocessing internally
        t0 = time.time()
        model_hybrid = HyperHybridOneStep() 
        model_hybrid.fit(X_train_raw, y_train)
        t_hybrid = time.time() - t0
        pred_hybrid = model_hybrid.predict(X_test_raw)
        results.append(("HyperHybrid", accuracy_score(y_test, pred_hybrid), f1_score(y_test, pred_hybrid, average='weighted'), t_hybrid))
        onestep_total_time += t_hybrid

        # PRINT RESULTS
        print(f"\n===== {name} =====")
        print(f"{'Model':<12} {'ACC':<8} {'F1':<8} {'Time':<8}")
        for r in results:
            # **************************************************
            # บรรทัดแก้ไข: แก้ไข format specifier จาก .4ff เป็น .4f
            # **************************************************
            print(f"{r[0]:<12} {r[1]:.4f}   {r[2]:.4f}   {r[3]:.4f}s")
            
    print(f"\nRAM End: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    
    if onestep_total_time > 0:
        speedup = xgb_total_time / onestep_total_time
    else:
        speedup = 0
        
    print("\n" + "="*70)
    print("AWAKEN vΩ.H — HYPER HYBRID ONESTEP CORE (Final Benchmark)")
    print(f"Total Speedup (XGB/HyperHybrid): {speedup:.1f}x")
    print("======================================================================")

if __name__ == "__main__":
    benchmark_hyper()
