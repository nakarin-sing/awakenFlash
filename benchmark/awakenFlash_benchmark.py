# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark_final_v4.py
THE DEFINITIVE CI-SAFE VERSION
- FIX: Added self.classes_ attribute to Poly2Wrapper/RFFWrapper to satisfy VotingClassifier.
- Preprocessing (StandardScaler) is done explicitly.
- Benchmark loop runs all three datasets.
"""

import numpy as np
import pandas as pd
import time
import tracemalloc

# =====================
# matplotlib safe import
# =====================
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    HAVE_MPL = True
except ModuleNotFoundError:
    print("matplotlib/seaborn not found, skipping plots")
    HAVE_MPL = False

# ----------------------
# sklearn + xgboost
# ----------------------
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, ClassifierMixin

# ----------------------
# Wrappers (FINAL FIX: Added self.classes_ attribute)
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
        # 💡 FINAL FIX: กำหนด classes_ attribute เพื่อให้ VotingClassifier ยอมรับ
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
        # 💡 FINAL FIX: กำหนด classes_ attribute เพื่อให้ VotingClassifier ยอมรับ
        self.classes_ = self.clf.classes_
        return self
    def predict(self, X):
        return self.clf.predict(self.rff.transform(X))
    def predict_proba(self, X):
        return self.clf.predict_proba(self.rff.transform(X))

# ----------------------
# Adaptive Hyperparameters (Used for consistent testing)
# ----------------------
def adaptive_hyperparameters(dataset_name):
    # ใช้ค่าที่ใกล้เคียงกับตรรกะเดิม
    if dataset_name == 'breast_cancer':
        return {'poly_C': 1.0, 'rff_C': 1.0, 'rff_gamma': 'scale', 'rff_n_components': 100}
    elif dataset_name == 'iris':
        return {'poly_C': 0.1, 'rff_C': 0.1, 'rff_gamma': 'scale', 'rff_n_components': 100}
    elif dataset_name == 'wine':
        return {'poly_C': 1.0, 'rff_C': 1.0, 'rff_gamma': 'scale', 'rff_n_components': 100}
    else:
        return {'poly_C': 1.0, 'rff_C': 1.0, 'rff_gamma': 'scale', 'rff_n_components': 100}


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
    
    # 2. Get Hyperparameters
    hparams = adaptive_hyperparameters(dataset_name)

    # 3. Define Models (NO PIPELINES)
    xgb_clf = xgb.XGBClassifier(max_depth=3, n_estimators=200,
                                use_label_encoder=False, eval_metric='logloss', random_state=42)
    poly2_clf = Poly2Wrapper(degree=2, C=hparams['poly_C'])
    rff_clf = RFFWrapper(gamma=hparams['rff_gamma'], n_components=hparams['rff_n_components'], C=hparams['rff_C'])

    # 4. Define Ensemble (using direct estimators)
    ensemble = VotingClassifier(
        estimators=[
            ('XGBoost', xgb_clf),
            ('Poly2', poly2_clf),
            ('RFF', rff_clf)
        ],
        voting='soft'
    )
    
    # 5. Model Dictionary
    models = {"XGBoost": xgb_clf, "Poly2": poly2_clf, "RFF": rff_clf, "Ensemble (Soft)": ensemble}
    
    # 6. Benchmark Loop
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        try:
            # ใช้ข้อมูลที่ถูก Scaled แล้วโดยตรง
            X_train_data = X_train_scaled
            X_test_data = X_test_scaled
            
            # CV safe: skip for Ensemble
            if "Ensemble" not in name:
                # Start memory & time tracking (for CV)
                tracemalloc.start()
                t_cv_start = time.time()
                cv_scores = cross_val_score(model, X_train_data, y_train, cv=cv, scoring='accuracy')
                t_cv_end = time.time()
                cv_mean_acc = cv_scores.mean()
                mem_current, mem_peak_cv = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                cv_time = t_cv_end - t_cv_start
            else:
                cv_mean_acc = np.nan
                mem_peak_cv = 0
                cv_time = 0 

            # Start memory & time tracking (for fit)
            tracemalloc.start()
            t0 = time.time()
            model.fit(X_train_data, y_train)
            train_time = time.time() - t0
            mem_current, mem_peak_fit = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Predict
            t0 = time.time()
            y_train_pred = model.predict(X_train_data)
            y_test_pred = model.predict(X_test_data)
            predict_time = time.time() - t0

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
                'Memory peak (MB)': round(mem_peak_fit/1e6, 4)
            })
        except Exception as e:
            print(f"Failed {name}: {e}")
            
    return pd.DataFrame(results)

# ----------------------
# Main Execution
# ----------------------
if __name__ == '__main__':
    all_results = []
    
    # Run Breast Cancer
    data_bc = load_breast_cancer()
    all_results.append(run_benchmark('breast_cancer', data_bc.data, data_bc.target))

    # Run Iris
    data_iris = load_iris()
    all_results.append(run_benchmark('iris', data_iris.data, data_iris.target))
    
    # Run Wine
    data_wine = load_wine()
    all_results.append(run_benchmark('wine', data_wine.data, data_wine.target))

    # Combine Results
    df_final = pd.concat(all_results, ignore_index=True)

    # ----------------------
    # FIX: ส่วนที่แสดงผลลัพธ์ใน Log อย่างสมบูรณ์
    # ----------------------
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n--- FINAL Benchmark Results (Full) ---\n")
    print(df_final.to_string())
    df_final.to_csv("benchmark_results.csv", index=False)

    # ----------------------
    # Optional plot
    # ----------------------
    if HAVE_MPL:
        # Plot Test ACC
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_final, x='Model', y='Test ACC', hue='Dataset')
        plt.title("Benchmark Test ACC by Dataset (Final Version)")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig("benchmark_test_acc_final.png", dpi=300)
        
        # Plot Memory
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_final, x='Model', y='Memory peak (MB)', hue='Dataset')
        plt.title("Memory Peak during Fit (MB)")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig("benchmark_memory_final.png", dpi=300)
