#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.16 — STREAMING MICRO-ENSEMBLE CORE
"Online OneStep+ with minimal memory usage and incremental ensemble voting"
MIT © 2025 xAI Research
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ========================================
# ONLINE ONESTEP CORE (Streaming)
# ========================================

class OnlineOneStep:
    """
    Incremental RLS with minimal quadratic + interaction terms.
    Train one sample at a time (or mini-batch) -> low RAM.
    """
    def __init__(self, n_features, n_classes, C=1e-3):
        self.n_classes = n_classes
        self.C = C
        
        # Feature expansion size: bias + n_features + quadratic + interactions
        self.n_orig = n_features
        self.n_feat = 1 + n_features + n_features + n_features*(n_features-1)//2
        
        # Initialize weights and P matrix for RLS
        self.W = np.zeros((self.n_feat, n_classes), dtype=np.float32)
        self.P = np.eye(self.n_feat, dtype=np.float32) / C

    def _expand(self, x):
        # x shape: (n_features,)
        feats = [1.0]  # bias
        feats.extend(x)  # linear
        feats.extend(x**2)  # quadratic
        # interactions
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                feats.append(x[i]*x[j])
        return np.array(feats, dtype=np.float32)  # shape (n_feat,)

    def partial_fit(self, X, y):
        for xi, yi in zip(X, y):
            phi = self._expand(xi).reshape(-1,1)  # column vector
            P_phi = self.P @ phi
            gain = P_phi / (1 + phi.T @ P_phi)
            y_onehot = np.zeros((self.n_classes,1), dtype=np.float32)
            y_onehot[yi,0] = 1.0
            self.W += (gain @ (y_onehot.T - phi.T @ self.W))
            self.P -= gain @ phi.T @ self.P

    def predict(self, X):
        X_exp = np.array([self._expand(x) for x in X], dtype=np.float32)
        return (X_exp @ self.W).argmax(axis=1)

# ========================================
# MICRO ENSEMBLE (Online Voting)
# ========================================

class StreamingMicroEnsemble:
    """
    Ensemble of N OnlineOneStep models with online incremental training.
    """
    def __init__(self, n_estimators=5, n_features=None, n_classes=None, C=1e-3):
        self.n_estimators = n_estimators
        self.models = [OnlineOneStep(n_features, n_classes, C) for _ in range(n_estimators)]

    def partial_fit(self, X, y):
        for m in self.models:
            m.partial_fit(X, y)

    def predict(self, X):
        votes = np.array([m.predict(X) for m in self.models])
        # Majority voting
        preds = []
        for i in range(X.shape[0]):
            counts = np.bincount(votes[:,i])
            preds.append(counts.argmax())
        return np.array(preds)

# ========================================
# BENCHMARK STREAMING SIMULATION
# ========================================

def benchmark_streaming():
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        n_features = X.shape[1]
        n_classes = y.max()+1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize streaming micro ensemble
        ensemble = StreamingMicroEnsemble(n_estimators=5, n_features=n_features, n_classes=n_classes, C=1e-3)
        
        # Train sample-by-sample (simulating streaming)
        for xi, yi in zip(X_train, y_train):
            ensemble.partial_fit(xi.reshape(1,-1), np.array([yi]))

        pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')

        print(f"\n===== {name} (Streaming MicroEnsemble) =====")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, RAM ~ minimal")

if __name__ == "__main__":
    benchmark_streaming()
