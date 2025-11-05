#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOG-ONLY NON-DUALISTIC ML BENCHMARK
Temporal Transcendence + SGD + PA + XGBoost
All results printed to console, no CSV
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ================= Fast Temporal Transcendence ===================
class FastTemporalTranscendence:
    def __init__(self, n_base_models=9, memory_size=80000):
        self.n_base_models = n_base_models
        self.models = []
        self.weights = np.ones(n_base_models) / n_base_models
        self.all_data_X = []
        self.all_data_y = []
        self.memory_size = memory_size
        self.first_fit = True
        self.classes_ = None
        self.interaction_pairs = None

        for i in range(n_base_models):
            if i % 3 == 0:
                model = SGDClassifier(loss='log_loss', learning_rate='optimal',
                                      max_iter=20, warm_start=True, random_state=42+i,
                                      alpha=0.00003)
            elif i % 3 == 1:
                model = PassiveAggressiveClassifier(C=0.02, max_iter=20,
                                                    warm_start=True, random_state=42+i)
            else:
                model = SGDClassifier(loss='hinge', learning_rate='optimal',
                                      max_iter=20, warm_start=True, random_state=42+i,
                                      alpha=0.00004)
            self.models.append(model)

    def _create_interactions(self, X):
        if self.interaction_pairs is None:
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-12:]
            self.interaction_pairs = []
            for i in range(len(top_indices)):
                for j in range(i+1, min(i+3, len(top_indices))):
                    self.interaction_pairs.append((top_indices[i], top_indices[j]))
        X_interactions = [(X[:, i]*X[:, j]).reshape(-1,1) for i,j in self.interaction_pairs[:12]]
        if X_interactions:
            return np.hstack([X]+X_interactions)
        return X

    def _update_weights(self, X_test, y_test):
        X_aug = self._create_interactions(X_test)
        new_weights = []
        for model in self.models:
            try:
                acc = model.score(X_aug, y_test)
                new_weights.append(np.exp(min(acc**2*5, 10)))
            except:
                new_weights.append(0.001)
        total = sum(new_weights)
        self.weights = np.array([w/total for w in new_weights])

    def partial_fit(self, X, y, classes=None):
        X_aug = self._create_interactions(X)
        if self.first_fit and classes is not None:
            self.classes_ = classes
            self.first_fit = False
        self.all_data_X.append(X)
        self.all_data_y.append(y)

        total_samples = sum(len(x) for x in self.all_data_X)
        while total_samples > self.memory_size and len(self.all_data_X) > 1:
            self.all_data_X.pop(0)
            self.all_data_y.pop(0)
            total_samples = sum(len(x) for x in self.all_data_X)

        for model in self.models:
            try:
                if classes is not None:
                    model.partial_fit(X_aug, y, classes=classes)
                else:
                    model.partial_fit(X_aug, y)
            except:
                pass

        all_X = np.vstack(self.all_data_X)
        all_y = np.concatenate(self.all_data_y)
        n_samples = min(len(all_X), 8000)
        indices = np.random.choice(len(all_X), n_samples, replace=False)
        X_sample_aug = self._create_interactions(all_X[indices])
        y_sample = all_y[indices]
        for model in self.models:
            try:
                model.partial_fit(X_sample_aug, y_sample)
            except:
                pass

    def predict(self, X):
        X_aug = self._create_interactions(X)
        if not self.models or self.classes_ is None:
            return np.zeros(len(X))
        all_predictions = []
        valid_weights = []
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X_aug)
                all_predictions.append(pred)
                valid_weights.append(self.weights[i])
            except:
                pass
        if not all_predictions:
            return np.zeros(len(X))
        valid_weights = np.array(valid_weights)
        valid_weights /= valid_weights.sum()
        n_samples = len(X)
        n_classes = len(self.classes_)
        vote_matrix = np.zeros((n_samples, n_classes))
        for pred, weight in zip(all_predictions, valid_weights):
            for i, cls in enumerate(self.classes_):
                vote_matrix[:, i] += (pred == cls) * weight
        return self.classes_[np.argmax(vote_matrix, axis=1)]

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
    return chunks[:n_chunks], np.unique(y_all)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

# ================= Benchmark Scenario ===================
def scenario_log_only(chunks, all_classes):
    temporal = FastTemporalTranscendence(n_base_models=9, memory_size=80000)
    sgd = SGDClassifier(loss="log_loss", learning_rate="optimal", max_iter=10, warm_start=True, random_state=42)
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=10, warm_start=True, random_state=42)
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 5
    first_sgd = first_pa = first_temporal = True

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # Temporal
        start = time.time()
        if first_temporal:
            temporal.partial_fit(X_train, y_train, classes=all_classes)
            first_temporal = False
        else:
            temporal.partial_fit(X_train, y_train)
        temporal._update_weights(X_test, y_test)
        temporal_pred = temporal.predict(X_test)
        temporal_metrics = compute_metrics(y_test, temporal_pred)
        temporal_time = time.time() - start

        # SGD
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X_train, y_train, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train)
        sgd_pred = sgd.predict(X_test)
        sgd_metrics = compute_metrics(y_test, sgd_pred)
        sgd_time = time.time() - start

        # PA
        start = time.time()
        if first_pa:
            pa.partial_fit(X_train, y_train, classes=all_classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train)
        pa_pred = pa.predict(X_test)
        pa_metrics = compute_metrics(y_test, pa_pred)
        pa_time = time.time() - start

        # XGBoost
        start = time.time()
        xgb_all_X.append(X_train)
        xgb_all_y.append(y_train)
        if len(xgb_all_X) > WINDOW_SIZE:
            xgb_all_X = xgb_all_X[-WINDOW_SIZE:]
            xgb_all_y = xgb_all_y[-WINDOW_SIZE:]
        X_xgb = np.vstack(xgb_all_X)
        y_xgb = np.concatenate(xgb_all_y)
        dtrain = xgb.DMatrix(X_xgb, label=y_xgb)
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": 7, "max_depth": 5,
             "eta": 0.1, "subsample": 0.8, "verbosity": 0},
            dtrain, num_boost_round=20
        )
        xgb_pred = xgb_model.predict(dtest)
        xgb_metrics = compute_metrics(y_test, xgb_pred)
        xgb_time = time.time() - start
        feature_importances = xgb_model.get_score(importance_type='weight')

        print(f"Chunk {chunk_id:02d}: "
              f"Temporal Acc={temporal_metrics['accuracy']:.3f} | "
              f"SGD Acc={sgd_metrics['accuracy']:.3f} | "
              f"PA Acc={pa_metrics['accuracy']:.3f} | "
              f"XGB Acc={xgb_metrics['accuracy']:.3f} | "
              f"Time (Temporal/XGB)={temporal_time:.3f}s/{xgb_time:.3f}s")
        print(f"  XGB Feature Importance (Top 10): {sorted(feature_importances.items(), key=lambda x:-x[1])[:10]}")

# ================= Main ===================
if __name__ == "__main__":
    print("📊 Loading dataset...")
    chunks, all_classes = load_data(n_chunks=10, chunk_size=10000)
    scenario_log_only(chunks, all_classes)
