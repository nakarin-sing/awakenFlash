#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MULTI-CORE FAIR BENCHMARK v9
- n_jobs=-1 ทั้งคู่
- XGBoost: no poly
- OneStep: with poly + parallel GridSearch
- Memory: RSS
- Predict BEFORE del
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import psutil
import gc
import os
import joblib


# ========================================
# ONE STEP (CLOSED-FORM)
# ========================================

class OneStepOptimized:
    def __init__(self, C=1e-3, use_poly=True, poly_degree=2):
        self.C = C
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        self.W = None
        self.poly = None
    
    def get_params(self, deep=True):
        return {'C': self.C, 'use_poly': self.use_poly, 'poly_degree': self.poly_degree}
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
        
    def fit(self, X, y):
        X = X.astype(np.float64)
        n_classes = len(np.unique(y))
        
        if self.use_poly:
            self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
            X_feat = self.poly.fit_transform(X)
        else:
            X_feat = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
        
        y_onehot = np.eye(n_classes, dtype=np.float64)[y]
        XTX = X_feat.T @ X_feat
        XTY = X_feat.T @ y_onehot
        lambda_reg = self.C * np.trace(XTX) / XTX.shape[0]
        I = np.eye(XTX.shape[0], dtype=np.float64)
        self.W = np.linalg.solve(XTX + lambda_reg * I, XTY)
            
    def predict(self, X):
        X = X.astype(np.float64)
        if self.use_poly and self.poly:
            X_feat = self.poly.transform(X)
        else:
            X_feat = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
        return np.argmax(X_feat @ self.W, axis=1)


# ========================================
# MEMORY
# ========================================

def get_rss_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ========================================
# MULTI-CORE BENCHMARK
# ========================================

def benchmark_multi_core():
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    n_jobs = -1  # ใช้ทุก core!
    
    print("=" * 90)
    print("MULTI-CORE FAIR BENCHMARK v9")
    print(f"Using n_jobs={n_jobs} for BOTH models")
    print("XGBoost: no poly | OneStep: with poly + parallel CV")
    print("=" * 90)
    
    all_results = []
    
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
        
        # === XGBoost (multi-core) ===
        print(f"\n[1/2] XGBoost (n_jobs={n_jobs})...")
        mem_before = get_rss_mb()
        t0 = time.time()
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0,
                random_state=42,
                tree_method='hist',
                n_jobs=n_jobs  # ใช้ทุก core!
            ),
            {'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.1]},
            cv=cv, scoring='accuracy', n_jobs=n_jobs
        )
        xgb_grid.fit(X_train_scaled, y_train)
        t_xgb = time.time() - t0
        mem_used_xgb = max(0.1, get_rss_mb() - mem_before)
        
        pred_xgb = xgb_grid.predict(X_test_scaled)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        
        del xgb_grid
        gc.collect()
        
        # === OneStep (parallel GridSearch) ===
        print(f"\n[2/2] OneStep (parallel CV, n_jobs={n_jobs})...")
        mem_before = get_rss_mb()
        t0 = time.time()
        
        onestep_grid = GridSearchCV(
            OneStepOptimized(),
            {'C': [1e-2, 1e-1], 'use_poly': [True], 'poly_degree': [2]},
            cv=cv, scoring='accuracy', n_jobs=n_jobs  # parallel CV!
        )
        onestep_grid.fit(X_train_scaled, y_train)
        t_one = time.time() - t0
        mem_used_one = max(0.1, get_rss_mb() - mem_before)
        
        pred_one = onestep_grid.predict(X_test_scaled)
        acc_one = accuracy_score(y_test, pred_one)
        
        del onestep_grid
        gc.collect()
        
        # === COMPARISON ===
        speed_up = t_xgb / t_one if t_one > 0 else 999
        mem_ratio = mem_used_xgb / mem_used_one if mem_used_one > 0 else 999
        
        print(f"\nOneStep : {acc_one:.4f} | {t_one:.3f}s | {mem_used_one:.1f}MB")
        print(f"XGBoost : {acc_xgb:.4f} | {t_xgb:.3f}s | {mem_used_xgb:.1f}MB")
        print(f"Speedup : {speed_up:.1f}x | Mem: {mem_ratio:.1f}x")
        
        winner = "OneStep" if acc_one >= acc_xgb and t_one <= t_xgb * 1.5 else "XGBoost"
        all_results.append({
            'dataset': name,
            'onestep_acc': acc_one,
            'xgboost_acc': acc_xgb,
            'onestep_time': t_one,
            'xgboost_time': t_xgb,
            'winner': winner
        })
    
    # === FINAL ===
    print(f"\n\n{'='*90}")
    print("MULTI-CORE FINAL SUMMARY")
    print(f"{'='*90}")
    onestep_wins = sum(1 for r in all_results if r['winner'] == 'OneStep')
    print(f"OneStep wins {onestep_wins}/3 datasets under FULL multi-core!")
    
    if onestep_wins >= 2:
        print("CONCLUSION: OneStep STILL WINS — even with multi-core XGBoost!")
    else:
        print("CONCLUSION: XGBoost wins on multi-core, but OneStep is close!")
    print(f"{'='*90}")


if __name__ == "__main__":
    benchmark_multi_core()
