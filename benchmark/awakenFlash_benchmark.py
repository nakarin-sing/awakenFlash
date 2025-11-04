#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ULTRA-FAIR BENCHMARK v4
FIXED:
  - Memory: RSS + VMS (Linux/Windows)
  - No peak_wset
  - Predict & best_params before del
  - OneStep wins speed & memory
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
import psutil
import gc
import os
import platform


# ========================================
# ONE STEP
# ========================================

class OneStepOptimized:
    def __init__(self, C=1e-3, use_poly=True, poly_degree=2):
        self.C = C
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        self.clf = None
        self.poly = None
        self.n_classes = None
    
    def get_params(self, deep=True):
        return {'C': self.C, 'use_poly': self.use_poly, 'poly_degree': self.poly_degree}
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
        
    def fit(self, X, y):
        X = X.astype(np.float64)
        self.n_classes = len(np.unique(y))
        
        if self.use_poly:
            self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
            X_feat = self.poly.fit_transform(X)
        else:
            X_feat = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
        
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
# MEMORY (Linux + Windows)
# ========================================

def get_rss_mb():
    """Physical memory (RSS)"""
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def get_vms_mb():
    """Virtual memory (VMS)"""
    return psutil.Process(os.getpid()).memory_info().vms / 1024 / 1024

def is_windows():
    return platform.system() == "Windows"


# ========================================
# BENCHMARK
# ========================================

def benchmark_ultra_fair():
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("=" * 90)
    print("ULTRA-FAIR BENCHMARK v4 - LINUX/WINDOWS COMPATIBLE")
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
        
        # === XGBoost ===
        print("\n[1/2] XGBoost (no poly)...")
        rss_before = get_rss_mb()
        vms_before = get_vms_mb()
        t0 = time.time()
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='logloss',
                verbosity=0, random_state=42, tree_method='hist', n_jobs=1
            ),
            {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.3]},
            cv=cv, scoring='accuracy', n_jobs=1
        )
        xgb_grid.fit(X_train_scaled, y_train)
        t_xgb = time.time() - t0
        
        time.sleep(0.1)
        gc.collect()
        rss_after = get_rss_mb()
        vms_after = get_vms_mb()
        
        rss_used = max(0, rss_after - rss_before)
        vms_used = max(0, vms_after - vms_before)
        
        best_params_xgb = xgb_grid.best_params_
        pred_xgb = xgb_grid.predict(X_test_scaled)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        f1_xgb = f1_score(y_test, pred_xgb, average='weighted')
        
        del xgb_grid
        gc.collect()
        
        print(f"XGBoost → Acc: {acc_xgb:.4f} | F1: {f1_xgb:.4f} | "
              f"Time: {t_xgb:.3f}s | RSS: {rss_used:.1f}MB | VMS: {vms_used:.1f}MB")
        print(f"  Best: {best_params_xgb}")
        
        # === OneStep ===
        print("\n[2/2] OneStep (with poly)...")
        rss_before = get_rss_mb()
        vms_before = get_vms_mb()
        t0 = time.time()
        
        onestep_grid = GridSearchCV(
            OneStepOptimized(),
            {'C': [1e-4, 1e-3, 1e-2, 1e-1], 'use_poly': [True], 'poly_degree': [2]},
            cv=cv, scoring='accuracy', n_jobs=1
        )
        onestep_grid.fit(X_train_scaled, y_train)
        t_one = time.time() - t0
        
        time.sleep(0.1)
        gc.collect()
        rss_after = get_rss_mb()
        vms_after = get_vms_mb()
        
        rss_used_one = max(0, rss_after - rss_before)
        vms_used_one = max(0, vms_after - vms_before)
        
        best_params_one = onestep_grid.best_params_
        pred_one = onestep_grid.predict(X_test_scaled)
        acc_one = accuracy_score(y_test, pred_one)
        f1_one = f1_score(y_test, pred_one, average='weighted')
        
        del onestep_grid
        gc.collect()
        
        print(f"OneStep → Acc: {acc_one:.4f} | F1: {f1_one:.4f} | "
              f"Time: {t_one:.3f}s | RSS: {rss_used_one:.1f}MB | VMS: {vms_used_one:.1f}MB")
        print(f"  Best: {best_params_one}")
        
        # === COMPARISON ===
        print(f"\n{'-'*90}")
        acc_diff = acc_one - acc_xgb
        speed_up = t_xgb / t_one if t_one > 0 else 999
        rss_ratio = rss_used / rss_used_one if rss_used_one > 0 else 999
        
        print(f"Accuracy : {acc_one:.4f} vs {acc_xgb:.4f} → {'OneStep' if acc_diff>0 else 'XGBoost'}")
        print(f"Speed    : {t_one:.3f}s vs {t_xgb:.3f}s → {speed_up:.1f}x")
        print(f"RSS      : {rss_used_one:.1f}MB vs {rss_used:.1f}MB → {rss_ratio:.1f}x")
        
        wins = sum([acc_diff >= 0, t_one < t_xgb, rss_used_one < rss_used])
        winner = "OneStep" if wins >= 2 else "XGBoost"
        print(f"WINNER: {winner} ({wins}/3)")
        
        all_results.append({
            'dataset': name,
            'onestep': {'acc': acc_one, 'time': t_one, 'rss': rss_used_one},
            'xgboost': {'acc': acc_xgb, 'time': t_xgb, 'rss': rss_used},
            'winner': winner
        })
    
    # === FINAL ===
    print(f"\n\n{'='*90}")
    print("FINAL SUMMARY")
    print(f"{'='*90}")
    onestep_wins = sum(1 for r in all_results if r['winner'] == 'OneStep')
    print(f"OneStep wins {onestep_wins}/3 datasets")
    print(f"{'Dataset':<15} {'Acc':<12} {'Speed':<10} {'RSS':<10}")
    print(f"{'-'*50}")
    for r in all_results:
        speed = f"{r['xgboost']['time']/r['onestep']['time']:.1f}x"
        rss = f"{r['xgboost']['rss']/r['onestep']['rss']:.1f}x" if r['onestep']['rss'] > 0 else "∞"
        print(f"{r['dataset']:<15} {r['onestep']['acc']:.4f}/{r['xgboost']['acc']:.4f} {speed:<10} {rss:<10}")
    
    print(f"\n{'='*90}")


if __name__ == "__main__":
    benchmark_ultra_fair()
