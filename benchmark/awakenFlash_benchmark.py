#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming Benchmark: OneStepStreaming (RLS) vs SGDClassifier
Goal: Test RLS core's ability to handle large data (simulated streaming)
      while maintaining high ACC and high speed (per-sample update).
MIT © 2025
"""

import time
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier # Baseline for streaming models
import tracemalloc
import psutil
import os
import gc

# ========================================
# AWAKEN vΩ.15: RLS STREAMING CORE
# ========================================

class OneStepStreaming(BaseEstimator, ClassifierMixin):
    """
    Recursive Least Squares (RLS) Core for Online/Streaming Learning
    Updates the weight matrix W iteratively for each incoming sample.
    """
    def __init__(self, C=1e-3, use_poly=True, poly_degree=2, forgetting_factor=1.0):
        # Parameters
        self.C = C
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        self.forgetting_factor = forgetting_factor
        
        # Model States
        self.W = None  # Weight Matrix
        self.P = None  # Inverse Covariance Matrix (P = (X^T X + λI)^-1)
        self.scaler = StandardScaler()
        self._poly = None
        self.n_classes = None
        
    def _transform_feature(self, X):
        """Applies Scaling and Polynomial features for a single sample or batch."""
        if self._poly is None:
             # Only fit Polynomial Features on the first call to determine N_features
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
            self._poly = poly.fit(X)
        
        X_features = self._poly.transform(X).astype(np.float32)
        return X_features
        
    def fit(self, X, y):
        """
        Fits the model by processing the dataset one sample at a time (Streaming).
        X must be pre-scaled (but Standard Scaler will be fitted).
        """
        # 0. Initial Preprocessing and Setup
        X = X.astype(np.float32)
        self.n_classes = y.max() + 1
        
        # 1. Fit Scaler (Simulates fitting on the first batch)
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Get Initial Feature Size
        X_phi_initial = self._transform_feature(X_scaled[0].reshape(1, -1))
        N_features = X_phi_initial.shape[1]
        
        # 3. Initialize W and P
        self.W = np.zeros((N_features, self.n_classes), dtype=np.float32)
        # Initialize P with a large value (inverse of regularization C)
        self.P = np.eye(N_features, dtype=np.float32) / self.C
        
        # 4. Streaming Loop (RLS Update)
        for i in range(X_scaled.shape[0]):
            x_i_scaled = X_scaled[i].reshape(1, -1)
            x_i_phi = self._transform_feature(x_i_scaled).T # Feature vector (N_features, 1)
            y_i = y[i]
            
            # One-hot encoding for the current sample
            y_i_oh = np.eye(self.n_classes, dtype=np.float32)[y_i].reshape(-1, 1) # (N_classes, 1)
            
            # --- RLS Update Formulas ---
            
            # 1. Prediction error (e_i)
            e_i = y_i_oh - (self.W.T @ x_i_phi) # (N_classes, 1)
            
            # 2. Gain vector (k_i) calculation
            denominator = self.forgetting_factor + x_i_phi.T @ self.P @ x_i_phi
            k_i = (self.P @ x_i_phi) / denominator
            
            # 3. Update Weight Matrix (W)
            self.W = self.W + k_i @ e_i.T
            
            # 4. Update Inverse Covariance Matrix (P)
            self.P = (self.P - k_i @ x_i_phi.T @ self.P) / self.forgetting_factor
            
        return self
            
    def predict(self, X):
        """Predict using learned weights W"""
        X = X.astype(np.float32)
        X_scaled = self.scaler.transform(X)
        
        # Apply same feature transformation
        X_features = self._transform_feature(X_scaled).astype(np.float32)
        
        # Compute predictions
        logits = X_features @ self.W
        return logits.argmax(axis=1)

# ========================================
# FAIR STREAMING BENCHMARK
# ========================================

def measure_memory():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_streaming_fair():
    """
    Fair streaming benchmark comparing RLS vs SGDClassifier.
    Note: Both use minimal resources per step, making them suitable for streaming.
    """
    
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]
    
    print("=" * 80)
    print("FAIR STREAMING BENCHMARK: RLS (vΩ.15) vs SGDClassifier")
    print("Comparison for Large Dataset / Online Learning Capability")
    print("=" * 80)
    
    for name, data in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {name.upper()} (Simulated Streaming)")
        print(f"{'='*80}")
        
        X, y = data.data, data.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # =====================================================================
        # 1. OneStepStreaming (RLS Core)
        # =====================================================================
        print("\n[1/2] Training OneStepStreaming (RLS)...")
        
        # Hyperparameter grid (Minimal GridSearch to find C)
        rls_params = {
            'C': [1e-4, 1e-3, 1e-2, 1e-1],
            'forgetting_factor': [1.0] # Use 1.0 for batch equivalent in RLS
        }
        
        tracemalloc.start()
        t0 = time.time()
        
        # GridSearchCV for RLS
        rls_grid = GridSearchCV(
            OneStepStreaming(),
            rls_params,
            cv=3,
            scoring='accuracy',
            n_jobs=1
        )
        # RLS must handle scaling internally for a fair streaming simulation
        rls_grid.fit(X_train, y_train)
        t_rls = time.time() - t0
        
        current_rls, peak_rls = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Predictions
        pred_rls = rls_grid.predict(X_test)
        acc_rls = accuracy_score(y_test, pred_rls)
        f1_rls = f1_score(y_test, pred_rls, average='weighted')
        
        results['RLS'] = {
            'accuracy': acc_rls, 'f1': f1_rls, 'time': t_rls,
            'peak_memory_mb': peak_rls / 1024 / 1024, 'best_params': rls_grid.best_params_
        }
        
        print(f"RLS Results (vΩ.15):")
        print(f"  Accuracy: {acc_rls:.4f}")
        print(f"  F1 Score: {f1_rls:.4f}")
        print(f"  Time (GridSearch): {t_rls:.4f}s")
        print(f"  Peak Memory: {peak_rls/1024/1024:.2f} MB")
        print(f"  Best Params: {rls_grid.best_params_}")
        
        # =====================================================================
        # 2. SGDClassifier (Streaming Baseline)
        # =====================================================================
        print("\n[2/2] Training SGDClassifier (Baseline)...")
        
        # SGD requires feature scaling, so we scale the whole set first for a fair comparison
        scaler_sgd = StandardScaler()
        X_train_scaled = scaler_sgd.fit_transform(X_train)
        X_test_scaled = scaler_sgd.transform(X_test)
        
        # SGD also benefits from polynomial features, but we keep it simpler for the baseline
        
        # Hyperparameter grid for SGD
        sgd_params = {
            'loss': ['log_loss'], 
            'penalty': ['l2', 'l1'],
            'alpha': [1e-4, 1e-3, 1e-2],
            'max_iter': [1000]
        }
        
        tracemalloc.start()
        t0 = time.time()
        
        sgd_grid = GridSearchCV(
            SGDClassifier(random_state=42, tol=1e-3, n_jobs=-1, early_stopping=True),
            sgd_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        sgd_grid.fit(X_train_scaled, y_train)
        t_sgd = time.time() - t0
        
        current_sgd, peak_sgd = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Predictions
        pred_sgd = sgd_grid.predict(X_test_scaled)
        acc_sgd = accuracy_score(y_test, pred_sgd)
        f1_sgd = f1_score(y_test, pred_sgd, average='weighted')
        
        results['SGD'] = {
            'accuracy': acc_sgd, 'f1': f1_sgd, 'time': t_sgd,
            'peak_memory_mb': peak_sgd / 1024 / 1024, 'best_params': sgd_grid.best_params_
        }
        
        print(f"SGD Results (Baseline):")
        print(f"  Accuracy: {acc_sgd:.4f}")
        print(f"  F1 Score: {f1_sgd:.4f}")
        print(f"  Time (GridSearch): {t_sgd:.4f}s")
        print(f"  Peak Memory: {peak_sgd/1024/1024:.2f} MB")
        print(f"  Best Params: {sgd_grid.best_params_}")
        
        # =====================================================================
        # 3. Comparison Summary
        # =====================================================================
        print(f"\n{'-'*80}")
        print("STREAMING COMPARISON SUMMARY (RLS vs SGD):")
        print(f"{'-'*80}")
        
        acc_diff = acc_rls - acc_sgd
        acc_winner = "RLS (vΩ.15)" if acc_diff > 1e-4 else "SGD" if acc_diff < -1e-4 else "TIE"
        print(f"Accuracy:    RLS {acc_rls:.4f} vs SGD {acc_sgd:.4f} → Winner: {acc_winner}")
        
        speedup = t_sgd / t_rls if t_rls > 0 else 0
        speed_winner = "RLS (vΩ.15)" if speedup > 1.05 else "SGD"
        print(f"Speed:       RLS {t_rls:.4f}s vs SGD {t_sgd:.4f}s → Speedup: {speedup:.2f}x → Winner: {speed_winner}")
        
        # Note: Peak memory is harder to compare in streaming mode as it depends on batch size
        
        print(f"\n{'*'*80}")
        print(f"CONCLUSION: {acc_winner} is the superior streaming core for ACC.")
        print(f"{'*'*80}")

if __name__ == "__main__":
    benchmark_streaming_fair()
