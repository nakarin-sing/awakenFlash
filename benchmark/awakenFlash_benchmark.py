#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ULTRA-FAIR BENCHMARK v5 - FINAL
FIXED:
  - Memory: min 0.1 MB
  - Overfit: poly_degree=[1,2], min_samples
  - Variance: Repeated CV
  - Table output
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RepeatedStratifiedKFold
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
import psutil
import gc
import os


# ========================================
# ONE STEP (ANTI-OVERFIT)
# ========================================

class OneStepOptimized:
    def __init__(self, C=1e-3, use_poly=True, poly_degree=2, min_samples=5):
        self.C = C
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        self.min_samples = min_samples
        self.clf = None
        self.poly = None
        self.n_classes = None
    
    def get_params(self, deep=True):
        return {
            'C': self.C, 'use_poly': self.use_poly,
            'poly_degree': self.poly_degree, 'min_samples': self.min_samples
        }
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
        
    def fit(self, X, y):
        X = X.astype(np.float64)
        n = X.shape[0]
        self.n_classes = len(np.unique(y))
        
        if self.use_poly:
            self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
            X_feat = self.poly.fit_transform(X)
        else:
            X_feat = np.hstack([np.ones((n, 1), dtype=np.float64), X])
        
        # Add small jitter to prevent singular matrix
        if X_feat.shape[1] > n // self.min_samples:
            X_feat = np.hstack([X_feat, np.random.randn(n, 1) * 1e-8])
        
        y_onehot = np.eye(self.n_classes, dtype=np.float64)[y]
        trace = np.trace(X_feat.T @ X_feat)
        alpha = self.C * trace / X_feat.shape[1] if X_feat.shape[1] > 0 else self.C
        
        self.clf = Ridge(alpha=alpha, solver='svd')
        self.clf.fit(X_feat, y_onehot)
            
    def predict(self, X):
        X = X.astype(np.float64)
        if self.use_poly and self.poly:
            X_feat = self.poly.transform(X)
        else:
            X_feat = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
        return np.argmax(self.clf.predict(X_feat), axis=1)


# ========================================
# MEMORY
# ========================================

def get_rss_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def safe_mem(delta):
    return max(0.1, delta)  # min 0.1 MB for small datasets


# ========================================
# BENCHMARK
# ========================================

def benchmark_ultra_fair():
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]
    
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)
    
    print("=" * 90)
    print("ULTRA-FAIR BENCHMARK v5 - FINAL")
    print("=" * 90)
    
    results_table = []
    
    for name, data in datasets:
        print(f"\n{'='*90}")
        print(f"DATASET: {name.upper()}")
        print(f"{'='*90}")
        
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # === XGBoost ===
        print("\n[1/2] XGBoost...")
        rss_before = get_rss_mb()
        t0 = time.time()
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='logloss',
                verbosity=0, random_state=42, tree_method='hist', n_jobs=1
            ),
            {'n_estimators': [50, 100], 'max_depth': [3], 'learning_rate': [0.1, 0.3]},
            cv=cv, scoring='accuracy', n_jobs=1
        )
        xgb_grid.fit(X_train_scaled, y_train)
        t_xgb = time.time() - t0
        
        time.sleep(0.1)
        gc.collect()
        rss_used_xgb = safe_mem(get_rss_mb() - rss_before)
        
        best_xgb = xgb_grid.best_params_
        pred_xgb = xgb_grid.predict(X_test_scaled)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        
        del xgb_grid; gc.collect()
        
        # === OneStep ===
        print("\n[2/2] OneStep...")
        rss_before = get_rss_mb()
        t0 = time.time()
        
        onestep_grid = GridSearchCV(
            OneStepOptimized(),
            {
                'C': [1e-3, 1e-2, 1e-1],
                'use_poly': [True, False],
                'poly_degree': [1, 2],
                'min_samples': [3, 5]
            },
            cv=cv, scoring='accuracy', n_jobs=1
        )
        onestep_grid.fit(X_train_scaled, y_train)
        t_one = time.time() - t0
        
        time.sleep(0.1)
        gc.collect()
        rss_used_one = safe_mem(get_rss_mb() - rss_before)
        
        best_one = onestep_grid.best_params_
        pred_one = onestep_grid.predict(X_test_scaled)
        acc_one = accuracy_score(y_test, pred_one)
        
        del onestep_grid; gc.collect()
        
        # === COMPARE ===
        speed_up = t_xgb / t_one if t_one > 0 else 999
        mem_ratio = rss_used_xgb / rss_used_one
        
        winner = "OneStep" if (acc_one >= acc_xgb and t_one < t_xgb) else "XGBoost"
        
        print(f"\nOneStep : {acc_one:.4f} | {t_one:.3f}s | {rss_used_one:.1f}MB | {best_one}")
        print(f"XGBoost : {acc_xgb:.4f} | {t_xgb:.3f}s | {rss_used_xgb:.1f}MB | {best_xgb}")
        print(f"WINNER: {winner} | Speed: {speed_up:.1f}x | Mem: {mem_ratio:.1f}x")
        
        results_table.append({
            'Dataset': name,
            'OneStep_Acc': f"{acc_one:.4f}",
            'XGBoost_Acc': f"{acc_xgb:.4f}",
            'Speed_x': f"{speed_up:.1f}",
            'Mem_x': f"{mem_ratio:.1f}",
            'Winner': winner
        })
    
    # === FINAL TABLE ===
    print(f"\n\n{'='*90}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*90}")
    print(f"{'Dataset':<15} {'OneStep':<10} {'XGBoost':<10} {'Speed':<8} {'Memory':<8} {'Winner':<8}")
    print(f"{'-'*68}")
    for r in results_table:
        print(f"{r['Dataset']:<15} {r['OneStep_Acc']:<10} {r['XGBoost_Acc']:<10} {r['Speed_x']:<8} {r['Mem_x']:<8} {r['Winner']:<8}")
    
    onestep_wins = sum(1 for r in results_table if r['Winner'] == 'OneStep')
    print(f"\nOneStep wins {onestep_wins}/3 datasets")
    print(f"{'='*90}")


if __name__ == "__main__":
    benchmark_ultra_fair()
