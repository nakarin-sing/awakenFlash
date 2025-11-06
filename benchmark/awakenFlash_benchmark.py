#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NON-LOGIC LIGHTNING BENCHMARK V2 - NaN Safe & Full Victory Pipeline
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============== Environment =================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ============== Feature Engine V2 =============
class NonLogicFeatureEngineV2:
    """Non-Logic Feature Engine V2: No NaN, stable numeric features"""
    def __init__(self, max_interactions=8, n_clusters=25):
        self.max_interactions = max_interactions
        self.n_clusters = n_clusters
        self.interaction_pairs = []
        self.kmeans = None
    
    def fit_transform(self, X):
        X = np.nan_to_num(X)  # No NaN
        n_features = X.shape[1]
        variances = np.var(X, axis=0)
        mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
        combined_importance = variances * (1 + mad)
        top_indices = np.argsort(combined_importance)[-6:]
        
        self.interaction_pairs = []
        for i in range(len(top_indices)):
            for j in range(i+1, min(i+3, len(top_indices))):
                if len(self.interaction_pairs) < self.max_interactions:
                    self.interaction_pairs.append((top_indices[i], top_indices[j]))
        
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=1000)
        cluster_features = self.kmeans.fit_transform(X)
        
        X_interactions = []
        for i, j in self.interaction_pairs:
            X_interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
            X_interactions.append(np.divide(X[:, i] + 1e-8, X[:, j] + 1e-8).reshape(-1, 1))
        
        all_features = [X, cluster_features]
        if X_interactions:
            all_features.append(np.hstack(X_interactions))
        
        return np.hstack(all_features)
    
    def transform(self, X):
        X = np.nan_to_num(X)
        cluster_features = self.kmeans.transform(X)
        X_interactions = []
        for i, j in self.interaction_pairs:
            X_interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
            X_interactions.append(np.divide(X[:, i] + 1e-8, X[:, j] + 1e-8).reshape(-1, 1))
        all_features = [X, cluster_features]
        if X_interactions:
            all_features.append(np.hstack(X_interactions))
        return np.hstack(all_features)

# ============== NonLogic Ensemble ============
class NonLogicEnsemble:
    """Diverse online ensemble with adaptive weighting"""
    def __init__(self, memory_size=15000):
        self.models = [
            SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=12, warm_start=True, random_state=42),
            PassiveAggressiveClassifier(C=0.05, max_iter=12, warm_start=True, random_state=43),
            SGDClassifier(loss='modified_huber', learning_rate='adaptive', max_iter=12, warm_start=True, random_state=44),
            PassiveAggressiveClassifier(C=0.08, max_iter=12, warm_start=True, random_state=45),
            SGDClassifier(loss='perceptron', learning_rate='optimal', max_iter=12, warm_start=True, random_state=46)
        ]
        self.weights = np.ones(len(self.models)) / len(self.models)
        self.all_data_X = []
        self.all_data_y = []
        self.memory_size = memory_size
        self.performance_history = []
        self.first_fit = True
        self.classes_ = None
        self.chunk_count = 0
    
    def _update_weights(self, X_val, y_val):
        perf = []
        for model in self.models:
            try:
                acc = max(0.1, model.score(X_val, y_val))
            except:
                acc = 0.1
            perf.append(acc)
        momentum = 0.4
        new_weights = momentum * self.weights + (1 - momentum) * np.array(perf)
        self.weights = new_weights / new_weights.sum()
    
    def partial_fit(self, X, y, classes=None):
        if self.first_fit and classes is not None:
            self.classes_ = classes
            self.first_fit = False
        self.chunk_count += 1
        
        self.all_data_X.append(X)
        self.all_data_y.append(y)
        total_samples = sum(len(x) for x in self.all_data_X)
        while total_samples > self.memory_size and len(self.all_data_X) > 2:
            self.all_data_X.pop(0)
            self.all_data_y.pop(0)
            total_samples = sum(len(x) for x in self.all_data_X)
        
        for model in self.models:
            try:
                if classes is not None:
                    model.partial_fit(X, y, classes=classes)
                else:
                    model.partial_fit(X, y)
            except:
                pass
        
        if len(self.all_data_X) >= 2 and self.chunk_count % 2 == 0:
            X_batch = np.vstack(self.all_data_X[-2:])
            y_batch = np.concatenate(self.all_data_y[-2:])
            n_samples = min(3000, len(X_batch))
            idx = np.random.choice(len(X_batch), n_samples, replace=False)
            for model in self.models:
                try:
                    model.partial_fit(X_batch[idx], y_batch[idx])
                except:
                    pass
    
    def predict(self, X):
        if not self.models or self.classes_ is None:
            return np.zeros(len(X))
        votes = np.zeros((len(X), len(self.classes_)))
        for model, w in zip(self.models, self.weights):
            try:
                pred = model.predict(X)
                for i, cls in enumerate(self.classes_):
                    votes[:, i] += (pred == cls) * w
            except:
                continue
        return self.classes_[np.argmax(votes, axis=1)]
    
    def score(self, X, y):
        pred = self.predict(X)
        acc = accuracy_score(y, pred)
        self._update_weights(X, y)
        self.performance_history.append(acc)
        if len(self.performance_history) > 5:
            self.performance_history.pop(0)
        return acc

# ============== Data Loader ==================
def load_data(n_chunks=5, chunk_size=5000):
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        df = pd.read_csv(url, header=None, nrows=60000)
    except:
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=60000, n_features=54, n_informative=25,
            n_redundant=15, n_classes=7, random_state=42
        )
        df = pd.DataFrame(X)
        df['target'] = y
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values
    y_all = y_all % 7
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, min(len(X_all), n_chunks*chunk_size), chunk_size)]
    return chunks, np.unique(y_all)

# ============== Benchmark ====================
def benchmark():
    chunks, classes = load_data(n_chunks=4, chunk_size=6000)
    feature_engine = NonLogicFeatureEngineV2(max_interactions=8, n_clusters=25)
    nonlogic = NonLogicEnsemble(memory_size=15000)
    sgd = SGDClassifier(loss="log_loss", learning_rate="optimal", max_iter=8, warm_start=True, random_state=42)
    pa = PassiveAggressiveClassifier(C=0.1, max_iter=8, warm_start=True, random_state=42)
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 3
    first_sgd = first_pa = first_nonlogic = True
    results = []

    # Fit feature engine
    if chunks:
        X_sample, _ = chunks[0]
        X_enhanced = feature_engine.fit_transform(X_sample[:2000])
        print(f"Enhanced features: {X_enhanced.shape[1]} (from {X_sample.shape[1]})")

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.7 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        X_train_eng = feature_engine.transform(X_train)
        X_test_eng = feature_engine.transform(X_test)
        print(f"\nChunk {chunk_id}/{len(chunks)} | Train: {len(X_train)}, Test: {len(X_test)}")

        # NonLogic
        start = time.time()
        if first_nonlogic:
            nonlogic.partial_fit(X_train_eng, y_train, classes=classes)
            first_nonlogic = False
        else:
            nonlogic.partial_fit(X_train_eng, y_train)
        nonlogic_pred = nonlogic.predict(X_test_eng)
        nonlogic_acc = accuracy_score(y_test, nonlogic_pred)
        nonlogic_time = time.time() - start

        # SGD & PA
        for model, first_flag, name in [(sgd, first_sgd, "SGD"), (pa, first_pa, "PA")]:
            start = time.time()
            if first_flag:
                model.partial_fit(X_train_eng, y_train, classes=classes)
                if name == "SGD": first_sgd = False
                else: first_pa = False
            else:
                model.partial_fit(X_train_eng, y_train)
            pred = model.predict(X_test_eng)
            acc = accuracy_score(y_test, pred)
            elapsed = time.time() - start
            print(f"  {name:10s}: {acc:.3f} ({elapsed:.2f}s)")

        # XGBoost
        xgb_all_X.append(X_train_eng)
        xgb_all_y.append(y_train)
        if len(xgb_all_X) > WINDOW_SIZE:
            xgb_all_X = xgb_all_X[-WINDOW_SIZE:]
            xgb_all_y = xgb_all_y[-WINDOW_SIZE:]
        X_xgb = np.vstack(xgb_all_X)
        y_xgb = np.concatenate(xgb_all_y)
        dtrain = xgb.DMatrix(X_xgb, label=y_xgb)
        dtest = xgb.DMatrix(X_test_eng, label=y_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax",
             "num_class": len(classes),
             "max_depth": 4,
             "eta": 0.1,
             "subsample": 0.8,
             "verbosity": 0,
             "nthread": 1},
            dtrain,
            num_boost_round=12
        )
        xgb_pred = xgb_model.predict(dtest)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        print(f"  XGB:       {xgb_acc:.3f}")

        results.append({
            'chunk': chunk_id,
            'nonlogic_acc': nonlogic_acc,
            'sgd_acc': sgd_acc,
            'pa_acc': pa_acc,
            'xgb_acc': xgb_acc
        })

    df_results = pd.DataFrame(results)
    print("\n📊 Benchmark Results:")
    print(df_results)
    return df_results

# ============== Main ========================
if __name__ == "__main__":
    start = time.time()
    df_results = benchmark()
    total_time = time.time() - start
    print(f"\nBenchmark completed in {total_time:.1f}s")
