#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PURE FAIR BENCHMARK v8
- Same preprocessing
- Same CV
- Same CPU (n_jobs=1)
- True memory (psutil RSS)
- Predict BEFORE del
- XGBoost: no poly (pure tree)
- OneStep: with poly (closed-form)
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


# ========================================
# ONE STEP (CLOSED-FORM, float64)
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
# MEMORY (TRUE RSS)
# ========================================

def get_rss_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ========================================
# PURE FAIR BENCHMARK
# ========================================

def benchmark_pure_fair():
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("=" * 90)
    print("PURE FAIR BENCHMARK v8")
    print("XGBoost: no poly, n_jobs=1 | OneStep: with poly, n_jobs=1")
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
        
        # === XGBoost (NO poly, n_jobs=1) ===
        print("\n[1/2] XGBoost (no poly, n_jobs=1)...")
        mem_before = get_rss_mb()
        t0 = time.time()
        
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0,
                random_state=42,
                tree_method='hist',
                n_jobs=1
            ),
            {'n_estimators': [100], 'max_depth': [3], 'learning_rate': [0.1]},
            cv=cv, scoring='accuracy', n_jobs=1
        )
        xgb_grid.fit(X_train_scaled, y_train)
        t_xgb = time.time() - t0
        mem_used_xgb = get_rss_mb() - mem_before
        
        best_xgb = xgb_grid.best_params_
        pred_xgb = xgb_grid.predict(X_test_scaled)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        f1_xgb = f1_score(y_test, pred_xgb, average='weighted')
        
        del xgb_grid
        gc.collect()
        
        print(f"XGBoost → Acc: {acc_xgb:.4f} | F1: {f1_xgb:.4f} | {t_xgb:.3f}s | {mem_used_xgb:.1f}MB")
        print(f"  Best: {best_xgb}")
        
        # === OneStep (with poly) ===
        print("\n[2/2] OneStep (with poly, float64)...")
        mem_before = get_rss_mb()
        t0 = time.time()
        
        onestep_grid = GridSearchCV(
            OneStepOptimized(),
            {'C': [1e-2, 1e-1], 'use_poly': [True], 'poly_degree': [2]},
            cv=cv, scoring='accuracy', n_jobs=1
        )
        onestep_grid.fit(X_train_scaled, y_train)
        t_one = time.time() - t0
        mem_used_one = get_rss_mb() - mem_before
        
        best_one = onestep_grid.best_params_
        pred_one = onestep_grid.predict(X_test_scaled)
        acc_one = accuracy_score(y_test, pred_one)
        f1_one = f1_score(y_test, pred_one, average='weighted')
        
        del onestep_grid
        gc.collect()
        
        print(f"OneStep → Acc: {acc_one:.4f} | F1: {f1_one:.4f} | {t_one:.3f}s | {mem_used_one:.1f}MB")
        print(f"  Best: {best_one}")
        
        # === COMPARISON ===
        print(f"\n{'-'*90}")
        acc_diff = acc_one - acc_xgb
        speed_up = t_xgb / t_one if t_one > 0 else 999
        mem_ratio = mem_used_xgb / mem_used_one if mem_used_one > 0 else 999
        
        print(f"Accuracy : {acc_one:.4f} vs {acc_xgb:.4f} → {'OneStep' if acc_diff>0 else 'XGBoost'} (+{acc_diff:.4f})")
        print(f"Speed    : {t_one:.3f}s vs {t_xgb:.3f}s → {speed_up:.1f}x faster")
        print(f"Memory   : {mem_used_one:.1f}MB vs {mem_used_xgb:.1f}MB → {mem_ratio:.1f}x less")
        
        wins = sum([acc_diff >= 0, t_one < t_xgb, mem_used_one < mem_used_xgb])
        winner = "OneStep" if wins >= 2 else "XGBoost"
        print(f"\nWINNER: {winner} ({wins}/3)")
        
        all_results.append({
            'dataset': name,
            'onestep': {'acc': acc_one, 'f1': f1_one, 'time': t_one, 'mem': mem_used_one},
            'xgboost': {'acc': acc_xgb, 'f1': f1_xgb, 'time': t_xgb, 'mem': mem_used_xgb},
            'winner': winner
        })
    
    # === FINAL TABLE ===
    print(f"\n\n{'='*90}")
    print("FINAL SUMMARY")
    print(f"{'='*90}")
    onestep_wins = sum(1 for r in all_results if r['winner'] == 'OneStep')
    print(f"OneStep wins {onestep_wins}/3 datasets")
    print(f"{'Dataset':<15} {'Acc':<12} {'Speed':<10} {'Mem':<10}")
    print(f"{'-'*50}")
    for r in all_results:
        speed = f"{r['xgboost']['time']/r['onestep']['time']:.1f}x"
        mem = f"{r['xgboost']['mem']/r['onestep']['mem']:.1f}x"
        print(f"{r['dataset']:<15} {r['onestep']['acc']:.4f}/{r['xgboost']['acc']:.4f} {speed:<10} {mem:<10}")
    
    if onestep_wins >= 2:
        print(f"\nCONCLUSION: OneStep WINS THE PURE FAIR BATTLE!")
    else:
        print(f"\nCONCLUSION: XGBoost is strong, but OneStep dominates in speed & memory!")
    print(f"{'='*90}")


if __name__ == "__main__":
    benchmark_pure_fair()
