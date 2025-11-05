#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRACTICAL NON-LOGIC ENSEMBLE - BEATING XGBOOST
Stable, fast, accurate tabular ensemble
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ================= Helper functions ===================
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, min(len(X_all), n_chunks * chunk_size), chunk_size)]
    return chunks, np.unique(y_all)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# ================= Practical Non-Logic Ensemble ===================
class PracticalEnsemble:
    def __init__(self):
        # Diverse but stable ensemble
        self.models = [
            RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1),
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42),
            LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='auto')
        ]
        self.ensemble = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.ensemble = VotingClassifier(
            estimators=[('rf', self.models[0]), ('gb', self.models[1]), ('lr', self.models[2])],
            voting='soft',  # ใช้ probability-weighted voting
            weights=[3, 4, 1],  # tree-based models ให้ weight สูงกว่า linear
            n_jobs=-1
        )
        self.ensemble.fit(X, y)

    def predict(self, X):
        return self.ensemble.predict(X)

# ================= Benchmark ===================
def benchmark(chunks, all_classes):
    practical_wins = 0
    practical_scores = []
    xgb_scores = []

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # Practical ensemble
        practical = PracticalEnsemble()
        practical.fit(X_train, y_train)
        practical_pred = practical.predict(X_test)
        practical_acc = compute_accuracy(y_test, practical_pred)
        practical_scores.append(practical_acc)

        # XGBoost baseline
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": len(all_classes), "max_depth": 8, "eta": 0.1,
             "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0},
            dtrain, num_boost_round=30
        )
        xgb_pred = xgb_model.predict(dtest)
        xgb_acc = compute_accuracy(y_test, xgb_pred)
        xgb_scores.append(xgb_acc)

        result = "🏆 PRACTICAL ENSEMBLE" if practical_acc > xgb_acc else "🎯 XGBOOST EDGE"
        if practical_acc > xgb_acc:
            practical_wins += 1

        print(f"Chunk {chunk_id:02d}: Practical={practical_acc:.3f} | XGBoost={xgb_acc:.3f} | {result}")

    print("\n=== Final Results ===")
    print(f"Average Practical Accuracy: {np.mean(practical_scores):.3f}")
    print(f"Average XGBoost Accuracy:   {np.mean(xgb_scores):.3f}")
    print(f"Practical wins: {practical_wins}/{len(chunks)} chunks")

# ================= Main ===================
if __name__ == "__main__":
    chunks, all_classes = load_data(n_chunks=10, chunk_size=10000)
    benchmark(chunks, all_classes)
