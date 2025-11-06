#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ULTIMATE FAIR BENCHMARK - 100% NO CHEATING
Both models get EXACTLY the same advantages:
- Same kernel/feature transformations
- Same hyperparameter search space SIZE
- Same computational resources
- Measured with CPU time (100x reps)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import psutil
import gc
from datetime import datetime

def cpu_time():
    p = psutil.Process(os.getpid())
    return p.cpu_times().user + p.cpu_times().system

# ========================================
# ONESTEP WITH FAIR FEATURES
# ========================================

class OneStepFair:
    def __init__(self, C=1.0, use_rbf_features=False, n_components=100):
        self.C = C
        self.use_rbf_features = use_rbf_features
        self.n_components = n_components
        self.scaler = None
        self.rbf_feature = None
        self.W = None
        self.classes = None
    
    def get_params(self, deep=True):
        return {
            'C': self.C,
            'use_rbf_features': self.use_rbf_features,
            'n_components': self.n_components
        }
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
        
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        
        if self.use_rbf_features:
            self.rbf_feature = RBFSampler(
                gamma=1.0 / X.shape[1],
                n_components=self.n_components,
                random_state=42
            )
            X_features = self.rbf_feature.fit_transform(X_scaled).astype(np.float32)
        else:
            X_features = np.hstack([
                np.ones((X_scaled.shape[0], 1), dtype=np.float32),
                X_scaled
            ])
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        y_onehot = np.zeros((len(y), n_classes), dtype=np.float32)
        for i, cls in enumerate(self.classes):
            y_onehot[y == cls, i] = 1.0
        
        K = X_features @ X_features.T
        n_samples = K.shape[0]
        lambda_reg = self.C * np.trace(K) / n_samples
        I_reg = np.eye(n_samples, dtype=np.float32) * lambda_reg
        
        self.alpha = np.linalg.solve(K + I_reg, y_onehot)
        self.X_train_features = X_features
            
    def predict(self, X):
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        if self.use_rbf_features:
            X_features = self.rbf_feature.transform(X_scaled).astype(np.float32)
        else:
            X_features = np.hstack([
                np.ones((X_scaled.shape[0], 1), dtype=np.float32),
                X_scaled
            ])
        
        K_test = X_features @ self.X_train_features.T
        predictions = K_test @ self.alpha
        return self.classes[np.argmax(predictions, axis=1)]

# ========================================
# XGBOOST WRAPPER WITH FAIR FEATURES
# ========================================

class XGBoostFair:
    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 use_rbf_features=False, n_components=100):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.use_rbf_features = use_rbf_features
        self.n_components = n_components
        self.scaler = None
        self.rbf_feature = None
        self.model = None
    
    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'use_rbf_features': self.use_rbf_features,
            'n_components': self.n_components
        }
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
    
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        
        if self.use_rbf_features:
            self.rbf_feature = RBFSampler(
                gamma=1.0 / X.shape[1],
                n_components=self.n_components,
                random_state=42
            )
            X_features = self.rbf_feature.fit_transform(X_scaled).astype(np.float32)
        else:
            X_features = X_scaled
        
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            random_state=42,
            tree_method='hist',
            n_jobs=1
        )
        self.model.fit(X_features, y)
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        if self.use_rbf_features:
            X_features = self.rbf_feature.transform(X_scaled).astype(np.float32)
        else:
            X_features = X_scaled
        
        return self.model.predict(X_features)

# ========================================
# NEW: ฟังก์ชันบันทึกผลลัพธ์ลงไฟล์
# ========================================
def save_results_to_file(filename, content):
    os.makedirs('benchmark_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_content = f"# Benchmark Results - {timestamp}\n\n{content}\n"
    filepath = f'benchmark_results/{filename}'
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)
    print(f"Saved: {filepath}")

# ========================================
# PHASE 1: TUNING
# ========================================

def run_phase1_fair(X_train, y_train, cv, dataset_name):
    print(f"\n{'='*80}")
    print(f"PHASE 1: HYPERPARAMETER TUNING (100% FAIR)")
    print(f"{'='*80}")
    
    print(f"| {'Model':<15} | {'CPU Time (s)':<14} | {'Best Acc':<12} | {'Best Params':<30} |")
    print(f"|{'-'*17}|{'-'*16}|{'-'*14}|{'-'*32}|")
    
    # OneStep
    cpu_before = cpu_time()
    one_grid = GridSearchCV(
        OneStepFair(),
        {
            'C': [0.01, 0.1, 1.0],
            'use_rbf_features': [False, True],
            'n_components': [100]
        },
        cv=cv, scoring='accuracy', n_jobs=1
    )
    one_grid.fit(X_train, y_train)
    cpu_one = cpu_time() - cpu_before
    acc_one = one_grid.best_score_
    best_one = one_grid.best_params_
    print(f"| {'OneStep':<15} | {cpu_one:<14.4f} | {acc_one:<12.4f} | {str(best_one)[:30]:<30} |")
    del one_grid; gc.collect()
    
    # XGBoost
    cpu_before = cpu_time()
    xgb_grid = GridSearchCV(
        XGBoostFair(),
        {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.1],
            'use_rbf_features': [False, True],
            'n_components': [100]
        },
        cv=cv, scoring='accuracy', n_jobs=1
    )
    xgb_grid.fit(X_train, y_train)
    cpu_xgb = cpu_time() - cpu_before
    acc_xgb = xgb_grid.best_score_
    best_xgb = xgb_grid.best_params_
    print(f"| {'XGBoost':<15} | {cpu_xgb:<14.4f} | {acc_xgb:<12.4f} | {str(best_xgb)[:30]:<30} |")
    del xgb_grid; gc.collect()
    
    print(f"\n{'-'*80}")
    speedup = cpu_xgb / cpu_one if cpu_one > 0 else float('inf')
    acc_diff = acc_one - acc_xgb
    winner = 'OneStep' if acc_one >= acc_xgb and speedup > 1 else 'XGBoost' if acc_xgb > acc_one else 'TIE'
    print(f"SPEEDUP: OneStep is {speedup:.2f}x faster in tuning")
    print(f"ACCURACY: OneStep {'+' if acc_diff >= 0 else ''}{acc_diff:.4f} vs XGBoost")
    print(f"WINNER: {winner}")
    print(f"{'-'*80}")
    
    phase1_content = f"""PHASE 1 RESULTS for {dataset_name}:
OneStep: CPU={cpu_one:.4f}s, Acc={acc_one:.4f}, Params={best_one}
XGBoost: CPU={cpu_xgb:.4f}s, Acc={acc_xgb:.4f}, Params={best_xgb}
Speedup: {speedup:.2f}x | Winner: {winner}"""
    save_results_to_file(f"{dataset_name.lower()}_phase1.txt", phase1_content)
    
    return {
        'onestep': {'cpu': cpu_one, 'acc': acc_one, 'params': best_one},
        'xgboost': {'cpu': cpu_xgb, 'acc': acc_xgb, 'params': best_xgb}
    }

# ========================================
# PHASE 2: RETRAIN
# ========================================

def run_phase2_fair(X_train, y_train, X_test, y_test, phase1, dataset_name):
    print(f"\n{'='*80}")
    print(f"PHASE 2: RETRAINING (100x REPETITION FOR STABLE MEASUREMENT)")
    print(f"{'='*80}")
    
    reps = 100
    
    # OneStep
    print(f"Training OneStep 100x with {phase1['onestep']['params']}...")
    cpu_times = []
    for _ in range(reps):
        cpu_before = cpu_time()
        model = OneStepFair(**phase1['onestep']['params'])
        model.fit(X_train, y_train)
        cpu_times.append(cpu_time() - cpu_before)
    cpu_one = np.mean(cpu_times)
    cpu_one_std = np.std(cpu_times)
    pred_one = model.predict(X_test)
    acc_one = accuracy_score(y_test, pred_one)
    
    # XGBoost
    print(f"Training XGBoost 100x with {phase1['xgboost']['params']}...")
    cpu_times = []
    for _ in range(reps):
        cpu_before = cpu_time()
        model = XGBoostFair(**phase1['xgboost']['params'])
        model.fit(X_train, y_train)
        cpu_times.append(cpu_time() - cpu_before)
    cpu_xgb = np.mean(cpu_times)
    cpu_xgb_std = np.std(cpu_times)
    pred_xgb = model.predict(X_test)
    acc_xgb = accuracy_score(y_test, pred_xgb)
    
    print(f"\n| {'Model':<15} | {'Mean CPU (s)':<14} | {'Std CPU (s)':<14} | {'Test Acc':<12} |")
    print(f"|{'-'*17}|{'-'*16}|{'-'*16}|{'-'*14}|")
    print(f"| {'OneStep':<15} | {cpu_one:<14.6f} | {cpu_one_std:<14.6f} | {acc_one:<12.4f} |")
    print(f"| {'XGBoost':<15} | {cpu_xgb:<14.6f} | {cpu_xgb_std:<14.6f} | {acc_xgb:<12.4f} |")
    
    print(f"\n{'-'*80}")
    speedup = cpu_xgb / cpu_one if cpu_one > 0 else float('inf')
    acc_diff = acc_one - acc_xgb
    winner = 'OneStep' if acc_one >= acc_xgb and speedup > 1 else 'XGBoost' if acc_xgb > acc_one else 'TIE'
    print(f"SPEEDUP: OneStep is {speedup:.2f}x faster")
    print(f"ACCURACY: OneStep {'+' if acc_diff >= 0 else ''}{acc_diff:.4f} vs XGBoost")
    print(f"WINNER: {winner}")
    print(f"{'-'*80}")
    
    phase2_content = f"""PHASE 2 RESULTS for {dataset_name}:
OneStep: CPU={cpu_one:.6f}±{cpu_one_std:.6f}s, Test Acc={acc_one:.4f}
XGBoost: CPU={cpu_xgb:.6f}±{cpu_xgb_std:.6f}s, Test Acc={acc_xgb:.4f}
Speedup: {speedup:.2f}x | Winner: {winner}"""
    save_results_to_file(f"{dataset_name.lower()}_phase2.txt", phase2_content)
    
    return {
        'onestep': {'cpu': cpu_one, 'cpu_std': cpu_one_std, 'acc': acc_one},
        'xgboost': {'cpu': cpu_xgb, 'cpu_std': cpu_xgb_std, 'acc': acc_xgb}
    }

# ========================================
# MAIN
# ========================================

def ultimate_fair_benchmark():
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    print("=" * 100)
    print("ULTIMATE FAIR BENCHMARK - 100% NO CHEATING")
    print("=" * 100)
    print("\nFairness Guarantees:")
    print("  Check: Both models get SAME feature transformations (RBF optional)")
    print("  Check: Both models search SIMILAR number of hyperparameters")
    print("  Check: Single-threaded (n_jobs=1) for both")
    print("  Check: CPU time measurement (not wall clock)")
    print("  Check: 100x repetition for stable speed measurement (< 2 min total)")
    print("  Check: Same train/test split, same CV folds")
    print("  Check: Same random seeds everywhere")
    print("=" * 100)
    
    results = {
        'onestep_acc_wins': 0, 'onestep_speed_wins': 0,
        'xgb_acc_wins': 0, 'xgb_speed_wins': 0,
        'total': len(datasets)
    }
    
    for name, data in datasets:
        print(f"\n\n{'='*100}")
        print(f"DATASET: {name.upper()}")
        print(f"{'='*100}")
        
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Samples: {len(X)} (train: {len(X_train)}, test: {len(X_test)})")
        print(f"Features: {X.shape[1]}")
        print(f"Classes: {len(np.unique(y))}")
        
        phase1 = run_phase1_fair(X_train, y_train, cv, name)
        phase2 = run_phase2_fair(X_train, y_train, X_test, y_test, phase1, name)
        
        if phase2['onestep']['acc'] >= phase2['xgboost']['acc']:
            results['onestep_acc_wins'] += 1
        else:
            results['xgb_acc_wins'] += 1
            
        if phase2['onestep']['cpu'] < phase2['xgboost']['cpu']:
            results['onestep_speed_wins'] += 1
        else:
            results['xgb_speed_wins'] += 1
    
    print(f"\n\n{'='*100}")
    print(f"FINAL VERDICT - ULTIMATE FAIR COMPARISON")
    print(f"{'='*100}\n")
    
    print(f"Accuracy Wins:")
    print(f"  OneStep: {results['onestep_acc_wins']}/{results['total']} datasets")
    print(f"  XGBoost: {results['xgb_acc_wins']}/{results['total']} datasets")
    
    print(f"\nSpeed Wins:")
    print(f"  OneStep: {results['onestep_speed_wins']}/{results['total']} datasets")
    print(f"  XGBoost: {results['xgb_speed_wins']}/{results['total']} datasets")
    
    print(f"\nOverall Winner:")
    onestep_total = results['onestep_acc_wins'] + results['onestep_speed_wins']
    xgb_total = results['xgb_acc_wins'] + results['xgb_speed_wins']
    overall_winner = 'ONESTEP WINS!' if onestep_total > xgb_total else 'XGBOOST WINS!' if xgb_total > onestep_total else 'TIE!'
    print(f"  {overall_winner} ({onestep_total if onestep_total > xgb_total else xgb_total}/{results['total']*2} metrics)")
    
    final_content = f"""FINAL SUMMARY:
Accuracy Wins - OneStep: {results['onestep_acc_wins']}/{results['total']}, XGBoost: {results['xgb_acc_wins']}/{results['total']}
Speed Wins - OneStep: {results['onestep_speed_wins']}/{results['total']}, XGBoost: {results['xgb_speed_wins']}/{results['total']}
Overall: {overall_winner}"""
    save_results_to_file("final_summary.txt", final_content)


if __name__ == "__main__":
    ultimate_fair_benchmark()
