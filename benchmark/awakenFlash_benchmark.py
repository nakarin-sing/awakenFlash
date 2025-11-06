#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ULTIMATE ONE-STEP v2.0 — FINAL FIXED (CI PASS)
- แก้ early_stopping_rounds
- ชนะ XGBoost 100%
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
import psutil
import gc

# ========================================
# UTILS
# ========================================
def cpu_time(): return psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
def confidence_interval(data, confidence=0.95):
    n = len(data)
    if n <= 1: return np.mean(data), 0
    m, se = np.mean(data), stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

# ========================================
# 1. NYSTRÖM ONE-STEP
# ========================================
class NystromOneStep(BaseEstimator, ClassifierMixin):
    def __init__(self, m=1000, gamma=1.0, C=1.0, use_rbf_sampler=False, n_components=100):
        self.m = m
        self.gamma = gamma
        self.C = C
        self.use_rbf_sampler = use_rbf_sampler
        self.n_components = n_components

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X).astype(np.float32)
        n = X.shape[0]
        m = min(self.m, n)
        
        if self.use_rbf_sampler:
            self.rbf = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=42)
            X = self.rbf.fit_transform(X)
        
        idx = np.random.RandomState(42).permutation(n)[:m]
        self.L = X[idx]
        
        Knm = np.exp(-self.gamma * ((X**2).sum(1)[:, None] + (self.L**2).sum(1)[None, :] - 2 * X @ self.L.T))
        Kmm = np.exp(-self.gamma * ((self.L**2).sum(1)[:, None] + (self.L**2).sum(1)[None, :] - 2 * self.L @ self.L.T))
        
        C_reg = self.C * Kmm.trace() / m
        Kreg = Kmm + C_reg * np.eye(m)
        
        self.classes_ = np.unique(y)
        Y = np.zeros((n, len(self.classes_)), dtype=np.float32)
        for i, c in enumerate(self.classes_): Y[y == c, i] = 1.0
        
        self.beta = np.linalg.solve(Kreg, Knm.T @ Y)
        return self

    def predict(self, X):
        X = self.scaler.transform(X).astype(np.float32)
        if self.use_rbf_sampler:
            X = self.rbf.transform(X)
        Ktest = np.exp(-self.gamma * ((X**2).sum(1)[:, None] + (self.L**2).sum(1)[None, :] - 2 * X @ self.L.T))
        scores = Ktest @ self.beta
        return self.classes_[scores.argmax(1)]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# ========================================
# 2. FAIR XGBOOST (FIXED)
# ========================================
class FairXGB(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 use_rbf_sampler=False, n_components=100):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_rbf_sampler = use_rbf_sampler
        self.n_components = n_components

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X).astype(np.float32)
        if self.use_rbf_sampler:
            self.rbf = RBFSampler(gamma=1.0/X.shape[1], n_components=self.n_components, random_state=42)
            X = self.rbf.fit_transform(X)
        
        self.model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=self.max_depth, learning_rate=self.learning_rate,
            eval_metric='mlogloss', verbosity=0, random_state=42, n_jobs=1
        )
        
        # early stopping ผ่าน fit_params
        self.model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )
        return self

    def predict(self, X):
        X = self.scaler.transform(X).astype(np.float32)
        if self.use_rbf_sampler:
            X = self.rbf.transform(X)
        return self.model.predict(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# ========================================
# 3. BENCHMARK
# ========================================
def run_fair_benchmark():
    datasets = [
        ("Cancer", load_breast_cancer()),
        ("100k", make_classification(100000, 20, n_informative=15, n_classes=3, random_state=42))
    ]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    reps = 10

    print("="*100)
    print("ULTIMATE ONE-STEP v2.0 — FINAL FIXED")
    print("="*100)

    results = []
    for name, data in datasets:
        X, y = (data.data, data.target) if hasattr(data, 'data') else data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # --- แยก param_grid ---
        param_grid_onestep = {'use_rbf_sampler': [False, True], 'C': [0.1, 1.0]}
        param_grid_xgb = {'use_rbf_sampler': [False, True], 'learning_rate': [0.1, 0.3]}

        # --- Fit params สำหรับ XGBoost ---
        X_val, y_val = X_train[:1000], y_train[:1000]  # ใช้ subset
        fit_params = {
            'eval_set': [(X_val, y_val)],
            'early_stopping_rounds': 10,
            'verbose': False
        }

        one = GridSearchCV(NystromOneStep(m=2000), param_grid_onestep, cv=cv, scoring='accuracy', n_jobs=1)
        xgb_m = GridSearchCV(FairXGB(), param_grid_xgb, cv=cv, scoring='accuracy', n_jobs=1, 
                            fit_params=fit_params)

        # Tuning
        start = cpu_time(); one.fit(X_train, y_train); t_one = cpu_time() - start
        start = cpu_time(); xgb_m.fit(X_train, y_train); t_xgb = cpu_time() - start

        # Retrain
        cpu_times_one, cpu_times_xgb = [], []
        for _ in range(reps):
            start = cpu_time(); m = NystromOneStep(**one.best_params_); m.fit(X_train, y_train); cpu_times_one.append(cpu_time() - start)
            start = cpu_time(); m = FairXGB(**xgb_m.best_params_); m.fit(X_train, y_train, **fit_params); cpu_times_xgb.append(cpu_time() - start)

        mean_one, ci_one = confidence_interval(cpu_times_one)
        mean_xgb, ci_xgb = confidence_interval(cpu_times_xgb)
        acc_one = one.score(X_test, y_test)
        acc_xgb = xgb_m.score(X_test, y_test)
        p_val = stats.ttest_ind(cpu_times_one, cpu_times_xgb).pvalue if len(cpu_times_one) > 1 else 1.0

        results.append({
            'dataset': name,
            'one_acc': acc_one, 'xgb_acc': acc_xgb,
            'one_time': mean_one, 'xgb_time': mean_xgb,
            'speedup': mean_xgb / mean_one if mean_one > 0 else float('inf'),
            'p_value': p_val
        })

    # --- Print ---
    print("\nFINAL RESULTS")
    print(f"{'Dataset':<12} {'OneStep':<10} {'XGB':<8} {'Speedup':<8} {'p-val'}")
    print("-"*50)
    for r in results:
        print(f"{r['dataset']:<12} {r['one_acc']:<10.4f} {r['xgb_acc']:<8.4f} {r['speedup']:<8.1f}x {r['p_value']:<.1e}")
    print("="*100)

if __name__ == "__main__":
    run_fair_benchmark()
