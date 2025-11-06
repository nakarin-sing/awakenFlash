#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRACTICAL ENSEMBLE 4.1 - Adaptive, Parallel, Dominating XGBoost
Fully optimized version
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ================= Helper ===================
def load_data(n_chunks=8, chunk_size=8000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, min(len(X_all), n_chunks * chunk_size), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# ================= Practical Ensemble 4.1 ===================
class PracticalEnsemble:
    def __init__(self):
        self.rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=1
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.12,
            max_depth=7,
            min_samples_split=4,
            min_samples_leaf=1,
            subsample=0.85,
            random_state=42
        )
        self.lr = LogisticRegression(
            max_iter=2000,
            C=0.8,
            solver='lbfgs',
            multi_class='multinomial',
            random_state=42
        )
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_poly = self.poly.fit_transform(X)

        # Parallel fitting with return assignment
        def fit_model(model, X_data):
            model.fit(X_data, y)
            return model

        models = [self.rf, self.gb, self.lr]
        X_list = [X, X, X_poly]
        fitted_models = Parallel(n_jobs=3)(delayed(fit_model)(m, xd) for m, xd in zip(models, X_list))
        self.rf, self.gb, self.lr = fitted_models

    def predict(self, X):
        X_poly = self.poly.transform(X)
        rf_proba = self.rf.predict_proba(X)
        gb_proba = self.gb.predict_proba(X)
        lr_proba = self.lr.predict_proba(X_poly)
        
        # Adaptive weighted voting
        weighted_proba = (2.5 * rf_proba + 3.5 * gb_proba + 1 * lr_proba) / 7.0
        return self.classes_[np.argmax(weighted_proba, axis=1)]

# ================= Robust XGBoost ===================
class RobustXGBoost:
    def __init__(self):
        self.model = None
        self.classes_ = None
        self.label_mapping = None
        
    def _safe_labels(self, y):
        unique_labels = np.unique(y)
        if len(unique_labels) != max(unique_labels)+1 or min(unique_labels) != 0:
            self.label_mapping = {old: new for new, old in enumerate(unique_labels)}
            return np.array([self.label_mapping[val] for val in y])
        self.label_mapping = None
        return y
        
    def _reverse_labels(self, y):
        if self.label_mapping:
            reverse_mapping = {v: k for k, v in self.label_mapping.items()}
            return np.array([reverse_mapping[val] for val in y])
        return y

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_safe = self._safe_labels(y)
        dtrain = xgb.DMatrix(X, label=y_safe)
        self.model = xgb.train({
            "objective": "multi:softmax",
            "num_class": len(self.classes_),
            "max_depth": 8,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "verbosity": 0
        }, dtrain, num_boost_round=25)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest).astype(int)
        return self._reverse_labels(predictions)

# ================= Benchmark ===================
def championship_benchmark(chunks, all_classes):
    ensemble_scores = []
    xgb_scores = []
    ensemble_times = []
    xgb_times = []
    ensemble_wins = 0

    print("🏆 PRACTICAL ENSEMBLE 4.1 vs XGBoost")
    print("="*60)

    for idx, (X_chunk, y_chunk) in enumerate(chunks, 1):
        print(f"\n🎯 Chunk {idx}/{len(chunks)}")
        split = int(0.8*len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # Ensemble
        start = time.time()
        ensemble = PracticalEnsemble()
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        t_ens = time.time()-start
        acc_ens = compute_accuracy(y_test, ensemble_pred)
        ensemble_scores.append(acc_ens)
        ensemble_times.append(t_ens)

        # XGBoost
        start = time.time()
        xgb_model = RobustXGBoost()
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        t_xgb = time.time()-start
        acc_xgb = compute_accuracy(y_test, xgb_pred)
        xgb_scores.append(acc_xgb)
        xgb_times.append(t_xgb)

        # Compare
        if acc_ens > acc_xgb + 0.005:
            result = "✅ ENSEMBLE DOMINATES"
            ensemble_wins += 1
        elif acc_ens > acc_xgb:
            result = "⚡ ENSEMBLE LEADS"
            ensemble_wins += 1
        else:
            result = "🔥 XGBOOST LEADS"

        print(f"E={acc_ens:.3f} | X={acc_xgb:.3f} | {result}")
        print(f"⏱ Ensemble: {t_ens:.2f}s | XGBoost: {t_xgb:.2f}s")

    # Final results
    print("\n" + "="*60)
    print("🎉 CHAMPIONSHIP FINAL RESULTS 🎉")
    print("="*60)
    print(f"Average Ensemble Accuracy: {np.mean(ensemble_scores):.3f}")
    print(f"Average XGBoost Accuracy:  {np.mean(xgb_scores):.3f}")
    print(f"Average Ensemble Time: {np.mean(ensemble_times):.2f}s")
    print(f"Average XGBoost Time:  {np.mean(xgb_times):.2f}s")
    print(f"Ensemble wins: {ensemble_wins}/{len(chunks)} chunks")
    print("="*60)

# ================= Main ===================
if __name__ == "__main__":
    chunks, all_classes = load_data(n_chunks=8, chunk_size=8000)
    print("🎉 PRACTICAL ENSEMBLE 4.1 ACTIVATED")
    print("💪 Adaptive + Parallel + Optimized for XGBoost Domination\n")
    championship_benchmark(chunks, all_classes)
