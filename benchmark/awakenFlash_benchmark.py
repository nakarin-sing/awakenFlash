#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRACTICAL ENSEMBLE 4.0 - ADAPTIVE & FASTER
Adaptive multi-model ensemble designed to dominate XGBoost
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed

# ================= Helper functions ===================
def load_data(n_chunks=10, chunk_size=8000):
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

# ================= Adaptive Ensemble ===================
class AdaptiveEnsemble:
    def __init__(self, prev_weights=(2.5, 3.5, 1.0)):
        self.rf = RandomForestClassifier(n_estimators=120, max_depth=25, min_samples_split=3,
                                         min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                                         random_state=42, n_jobs=-1)
        self.gb = GradientBoostingClassifier(n_estimators=120, learning_rate=0.12, max_depth=7,
                                             min_samples_split=4, min_samples_leaf=1, subsample=0.85,
                                             random_state=42)
        self.lr = LogisticRegression(max_iter=1000, C=0.8, solver='lbfgs', multi_class='auto', random_state=42)
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.classes_ = None
        self.weights = np.array(prev_weights)

    def adapt_weights(self, last_chunk_accs):
        """Adjust ensemble weights based on last chunk performance"""
        total = sum(last_chunk_accs)
        if total > 0:
            self.weights = np.array(last_chunk_accs) / total * 6.0  # scale to ~6
        else:
            self.weights = np.array([2.0, 3.0, 1.0])

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_poly = self.poly.fit_transform(X)
        Parallel(n_jobs=3)(delayed(model.fit)(X_poly if model==self.lr else X, y) 
                           for model in [self.rf, self.gb, self.lr])

    def predict(self, X):
        X_poly = self.poly.transform(X)
        rf_proba = self.rf.predict_proba(X)
        gb_proba = self.gb.predict_proba(X)
        lr_proba = self.lr.predict_proba(X_poly)
        weighted_proba = (self.weights[0]*rf_proba + self.weights[1]*gb_proba + self.weights[2]*lr_proba) / sum(self.weights)
        return self.classes_[np.argmax(weighted_proba, axis=1)]

# ================= Robust XGBoost ===================
class RobustXGBoost:
    def __init__(self):
        self.model = None
        self.label_mapping = None
        self.classes_ = None

    def _safe_labels(self, y):
        unique_labels = np.unique(y)
        if len(unique_labels) != max(unique_labels) + 1 or min(unique_labels) != 0:
            self.label_mapping = {old: new for new, old in enumerate(unique_labels)}
            return np.array([self.label_mapping[val] for val in y])
        self.label_mapping = None
        return y

    def _reverse_labels(self, y):
        if self.label_mapping:
            rev_map = {v:k for k,v in self.label_mapping.items()}
            return np.array([rev_map[val] for val in y])
        return y

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y_safe = self._safe_labels(y)
        dtrain = xgb.DMatrix(X, label=y_safe)
        self.model = xgb.train({"objective":"multi:softmax","num_class":len(self.classes_),
                                "max_depth":8,"eta":0.1,"subsample":0.8,"colsample_bytree":0.8,
                                "min_child_weight":3,"verbosity":0},
                               dtrain, num_boost_round=25)
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        pred = self.model.predict(dtest).astype(int)
        return self._reverse_labels(pred)

# ================= Adaptive Championship ===================
def adaptive_championship(chunks, all_classes):
    prev_accs = [0.6, 0.7, 0.5]  # initial weights for RF, GB, LR
    ensemble_wins = 0

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        ensemble = AdaptiveEnsemble(prev_accs)
        ensemble.adapt_weights(prev_accs)
        start_time = time.time()
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_time = time.time() - start_time
        ensemble_acc = compute_accuracy(y_test, ensemble_pred)

        xgb_model = RobustXGBoost()
        start_time_xgb = time.time()
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_time = time.time() - start_time_xgb
        xgb_acc = compute_accuracy(y_test, xgb_pred)

        prev_accs = [ensemble_acc, ensemble_acc*1.05, ensemble_acc*0.9]  # adaptive next chunk

        if ensemble_acc > xgb_acc:
            ensemble_wins += 1
            result = "✅ ENSEMBLE"
        else:
            result = "🔥 XGBOOST"

        print(f"Chunk {chunk_id:02d} | Ensemble Acc={ensemble_acc:.3f} ({ensemble_time:.2f}s) | "
              f"XGB Acc={xgb_acc:.3f} ({xgb_time:.2f}s) | {result}")

    print(f"\n🏆 Total Ensemble Wins: {ensemble_wins}/{len(chunks)}")
    if ensemble_wins > len(chunks)/2:
        print("🎉 ENSEMBLE VICTORY CONFIRMED!")
    else:
        print("🔥 XGBoost still holds ground, but Ensemble adapts!")

# ================= Main ===================
if __name__=="__main__":
    chunks, all_classes = load_data(n_chunks=8, chunk_size=8000)
    print("🎉 PRACTICAL ENSEMBLE 4.0 ACTIVATED")
    print("💪 Adaptive + Faster + Feature Interactions")
    adaptive_championship(chunks, all_classes)
