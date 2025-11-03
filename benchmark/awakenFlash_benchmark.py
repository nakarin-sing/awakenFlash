# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark_final_v8.py
THE NON-LOGIC CONTEXTUAL VERSION (3-NON PRINCIPLE)
- NEW: Implemented a 'BenchmarkContext' class to wrap time and memory tracking.
- This adheres to the 3-Non principle (Non-Non-Non-Logic) by cleanly separating 
  the measurement process from the core logic, reducing "contextual attachment."
"""

import numpy as np
import pandas as pd
import time
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# sklearn + xgboost imports (remains the same)
# ----------------------
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------
# Wrappers (Remain the same)
# ----------------------
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier" 
    
    def __init__(self, degree=2, C=0.1):
        self.degree = degree
        self.C = C
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.clf = LogisticRegression(C=self.C, max_iter=5000, random_state=42)
    def fit(self, X, y):
        X_transformed = self.poly.fit_transform(X)
        self.clf.fit(X_transformed, y)
        self.classes_ = self.clf.classes_ 
        return self
    def predict(self, X):
        return self.clf.predict(self.poly.transform(X))
    def predict_proba(self, X):
        return self.clf.predict_proba(self.poly.transform(X))

class RFFWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, gamma='scale', n_components=100, C=1.0):
        self.gamma = gamma
        self.n_components = n_components
        self.C = C
        self.rff = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=42)
        self.clf = LogisticRegression(C=self.C, max_iter=5000, random_state=42)
    def fit(self, X, y):
        X_transformed = self.rff.fit_transform(X)
        self.clf.fit(X_transformed, y)
        self.classes_ = self.clf.classes_
        return self
    def predict(self, X):
        return self.clf.predict(self.rff.transform(X))
    def predict_proba(self, X):
        return self.clf.predict_proba(self.rff.transform(X))

# ----------------------
# Adaptive Hyperparameters and FLOPs Estimation (Remain the same)
# ----------------------
def adaptive_hyperparameters(dataset_name):
    if dataset_name == 'breast_cancer':
        return {'poly_C': 1.0, 'rff_C': 1.0, 'lr_C': 1.0, 'rff_gamma': 'scale', 'rff_n_components': 100}
    elif dataset_name == 'iris':
        return {'poly_C': 0.1, 'rff_C': 0.1, 'lr_C': 0.1, 'rff_gamma': 'scale', 'rff_n_components': 100}
    elif dataset_name == 'wine':
        return {'poly_C': 1.0, 'rff_C': 1.0, 'lr_C': 1.0, 'rff_gamma': 'scale', 'rff_n_components': 100}
    else:
        return {'poly_C': 1.0, 'rff_C': 1.0, 'lr_C': 1.0, 'rff_gamma': 'scale', 'rff_n_components': 100}

def estimate_flops(model, X_train, y_train, X_test, name):
    N, D = X_train.shape
    M = X_test.shape[0]
    flops = 0
    
    if "LogReg (Plain)" in name:
        flops = D * 2 + 4
    elif "Poly2" in name:
        X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_train[:1])
        D_transformed = X_poly.shape[1]
        flops = D_transformed * 2 + 4
    elif "RFF" in name:
        N_comp = model.n_components
        flops = N_comp * D * 2 + N_comp * 2 + 4
        
    return flops

# ----------------------
# 💡 NEW: Benchmark Context Manager
# ----------------------
class BenchmarkContext:
    """A context manager to track time and memory peak accurately."""
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.start_time = None
        self.peak_memory = 0
        self.duration = 0

    def __enter__(self):
        self.start_time = time.time()
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory = peak
        self.duration = self.end_time - self.start_time
        # No need to print error here; the main loop handles exceptions.

# ----------------------
# The Benchmark Function
# ----------------------
def run_benchmark(dataset_name, X, y):
    print(f"\n--- Running Benchmark for {dataset_name.upper()} ---")
    
    # 1. Split and Scale Data Explicitly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2-5. Model Definitions (Remain the same)
    hparams = adaptive_hyperparameters(dataset_name)
    xgb_clf = xgb.XGBClassifier(max_depth=3, n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)
    poly2_clf = Poly2Wrapper(degree=2, C=hparams['poly_C'])
    rff_clf = RFFWrapper(gamma=hparams['rff_gamma'], n_components=hparams['rff_n_components'], C=hparams['rff_C'])
    lr_clf = LogisticRegression(C=hparams['lr_C'], max_iter=5000, random_state=42) 
    ensemble = VotingClassifier(estimators=[('XGBoost', xgb_clf), ('LogReg', lr_clf)], voting='hard')
    
    models = {"XGBoost": xgb_clf, "Poly2": poly2_clf, "RFF": rff_clf, "LogReg (Plain)": lr_clf, "Ensemble (XGB+LR)": ensemble}
    
    # 6. Benchmark Loop
    cv = StratifiedKfold(n_splits=5, shuffle=True, random_state=42)
    results = []
    N_train = X_train_scaled.shape[0]
    N_test = X_test_scaled.shape[0]

    for name, model in models.items():
        try:
            X_train_data = X_train_scaled
            X_test_data = X_test_scaled
            
            # 1. Benchmarking CV (Using Context Manager)
            if "Ensemble" not in name:
                with BenchmarkContext(f"{name} CV") as ctx_cv:
                    cv_scores = cross_val_score(model, X_train_data, y_train, cv=cv, scoring='accuracy')
                cv_mean_acc = cv_scores.mean()
                cv_time = ctx_cv.duration
            else:
                cv_mean_acc = np.nan
                cv_time = 0 

            # 2. Benchmarking Fit (Using Context Manager)
            with BenchmarkContext(f"{name} Fit") as ctx_fit:
                model.fit(X_train_data, y_train)
            train_time = ctx_fit.duration
            mem_peak_fit = ctx_fit.peak_memory

            # 3. Benchmarking Predict (Using Context Manager)
            with BenchmarkContext(f"{name} Predict") as ctx_predict:
                y_train_pred = model.predict(X_train_data)
                y_test_pred = model.predict(X_test_data)
            predict_time = ctx_predict.duration

            # Calculate Speed Metrics and FLOPs
            train_speed = N_train / train_time if train_time > 0 else np.inf
            predict_speed = N_test / predict_time if predict_time > 0 else np.inf
            flops_per_sample = estimate_flops(model, X_train_scaled, y_train, X_test_scaled, name)
            
            # Save results
            results.append({
                'Dataset': dataset_name,
                'Model': name,
                'Train ACC': accuracy_score(y_train, y_train_pred),
                'Test ACC': accuracy_score(y_test, y_test_pred),
                'Train F1': f1_score(y_train, y_train_pred, average='weighted'),
                'Test F1': f1_score(y_test, y_test_pred, average='weighted'),
                'CV mean ACC': cv_mean_acc,
                'Train time (s)': round(train_time, 4),
                'Predict time (s)': round(predict_time, 4),
                'CV time (s)': round(cv_time, 4),
                'Memory peak (MB)': round(mem_peak_fit/1e6, 4),
                'Train Speed (samp/s)': round(train_speed, 1),
                'Predict Speed (samp/s)': round(predict_speed, 1),
                'Pred FLOPs/sample': flops_per_sample
            })
        except Exception as e:
            print(f"Failed {name}: {e}")
            
    return pd.DataFrame(results)

# ----------------------
# Main Execution (Remains the same)
# ----------------------
if __name__ == '__main__':
    all_results = []
    
    # Run all datasets
    data_bc = load_breast_cancer()
    all_results.append(run_benchmark('breast_cancer', data_bc.data, data_bc.target))
    data_iris = load_iris()
    all_results.append(run_benchmark('iris', data_iris.data, data_iris.target))
    data_wine = load_wine()
    all_results.append(run_benchmark('wine', data_wine.data, data_wine.target))

    # Combine Results
    df_final = pd.concat(all_results, ignore_index=True)

    # พิมพ์ผลลัพธ์ใน Log อย่างสมบูรณ์
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n--- FINAL Benchmark Results (Full) ---\n")
    print(df_final.to_string())
    df_final.to_csv("benchmark_results.csv", index=False)
