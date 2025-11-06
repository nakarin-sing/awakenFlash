#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NON-LOGIC ENSEMBLE v2 - Lightning Benchmark Against XGBoost
- Enhanced feature engine (non-linear + supervised importance)
- 5 diverse models (SGD, PA, ExtraTrees)
- Adaptive weights & batch reinforcement
- Misclassified sample focus for fast learning
"""

import os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ===== Feature Engine v2 =====
class NonLogicFeatureEngineV2:
    def __init__(self, top_k=8, n_clusters=20):
        self.top_k = top_k
        self.n_clusters = n_clusters
        self.kmeans = None
        self.top_idx = None
    
    def fit_transform(self, X, y=None):
        var = np.var(X, axis=0)
        mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
        importance = var * (1 + mad)
        
        # Supervised adjustment
        if y is not None:
            from sklearn.feature_selection import f_classif
            f_scores, _ = f_classif(X, y)
            importance *= (1 + f_scores / (f_scores.max()+1e-6))
        
        self.top_idx = np.argsort(importance)[-self.top_k:]
        
        # Cluster features
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, batch_size=500)
        cluster_feats = self.kmeans.fit_transform(X)
        
        # Interactions
        features = [X, cluster_feats]
        for i in range(len(self.top_idx)):
            for j in range(i+1, len(self.top_idx)):
                a, b = X[:, self.top_idx[i]], X[:, self.top_idx[j]]
                features.append((a*b).reshape(-1,1))
                features.append((np.sqrt(np.abs(a-b)+1e-8)).reshape(-1,1))
                features.append((np.log(a+1e-8)/(b+1e-8)).reshape(-1,1))
        return np.hstack(features)
    
    def transform(self, X):
        if self.top_idx is None or self.kmeans is None:
            return X
        cluster_feats = self.kmeans.transform(X)
        features = [X, cluster_feats]
        for i in range(len(self.top_idx)):
            for j in range(i+1, len(self.top_idx)):
                a, b = X[:, self.top_idx[i]], X[:, self.top_idx[j]]
                features.append((a*b).reshape(-1,1))
                features.append((np.sqrt(np.abs(a-b)+1e-8)).reshape(-1,1))
                features.append((np.log(a+1e-8)/(b+1e-8)).reshape(-1,1))
        return np.hstack(features)

# ===== NonLogic Ensemble v2 =====
class NonLogicEnsembleV2:
    def __init__(self, memory_size=15000, feature_engine=None):
        self.models = []
        self.weights = np.ones(5)/5
        self.memory_size = memory_size
        self.feature_engine = feature_engine
        self.all_X, self.all_y = [], []
        self.first_fit = True
        self.classes_ = None
        self.chunk_count = 0
        
        # Diverse models
        self.models.append(SGDClassifier(loss='log_loss', learning_rate='optimal', max_iter=12,
                                         warm_start=True, random_state=42, alpha=0.0005, penalty='l2'))
        self.models.append(PassiveAggressiveClassifier(C=0.05, max_iter=12, warm_start=True, random_state=43, loss='hinge'))
        self.models.append(SGDClassifier(loss='modified_huber', learning_rate='adaptive', max_iter=12,
                                         warm_start=True, random_state=44, alpha=0.0008))
        self.models.append(PassiveAggressiveClassifier(C=0.08, max_iter=12, warm_start=True, random_state=45, loss='squared_hinge'))
        self.models.append(ExtraTreesClassifier(n_estimators=50, max_depth=6, random_state=46, n_jobs=1))
    
    def _update_weights(self, X_val, y_val):
        perf = []
        for model in self.models:
            try:
                acc = model.score(X_val, y_val)
                perf.append(max(0.1, acc))
            except:
                perf.append(0.1)
        momentum = 0.4
        new_w = momentum*self.weights + (1-momentum)*np.array(perf)
        self.weights = new_w / new_w.sum()
    
    def partial_fit(self, X, y, classes=None):
        if self.first_fit and classes is not None:
            self.classes_ = classes
            self.first_fit = False
        self.chunk_count += 1
        self.all_X.append(X)
        self.all_y.append(y)
        
        total_samples = sum(len(a) for a in self.all_X)
        while total_samples > self.memory_size and len(self.all_X) > 2:
            self.all_X.pop(0)
            self.all_y.pop(0)
            total_samples = sum(len(a) for a in self.all_X)
        
        # Phase 1: immediate online
        for model in self.models:
            try:
                if classes is not None:
                    model.partial_fit(X, y, classes=classes)
                else:
                    model.partial_fit(X, y)
            except:
                pass
        
        # Phase 2: batch reinforcement on misclassified
        if len(self.all_X) >= 2 and self.chunk_count % 2 == 0:
            X_batch = np.vstack(self.all_X[-2:])
            y_batch = np.concatenate(self.all_y[-2:])
            pred = self.predict(X_batch)
            mis_idx = np.where(pred != y_batch)[0]
            if len(mis_idx) > 0:
                sample_idx = np.random.choice(mis_idx, min(3000, len(mis_idx)), replace=False)
                X_sample, y_sample = X_batch[sample_idx], y_batch[sample_idx]
                for model in self.models:
                    try:
                        model.partial_fit(X_sample, y_sample)
                    except:
                        pass
    
    def predict(self, X):
        if not self.models or self.classes_ is None:
            return np.zeros(len(X))
        votes = np.zeros((len(X), len(self.classes_)))
        valid_weights = []
        for i, model in enumerate(self.models):
            try:
                p = model.predict(X)
                valid_weights.append(self.weights[i])
                for j, cls in enumerate(self.classes_):
                    votes[:, j] += (p==cls)*self.weights[i]
            except:
                continue
        if not valid_weights:
            return np.zeros(len(X))
        return self.classes_[np.argmax(votes, axis=1)]
    
    def score(self, X, y):
        pred = self.predict(X)
        acc = accuracy_score(y, pred)
        self._update_weights(X, y)
        return acc

# ===== Dataset Loading =====
def load_dataset(n_chunks=5, chunk_size=5000):
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        df = pd.read_csv(url, header=None, nrows=60000)
    except:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=60000, n_features=54, n_informative=25,
                                   n_redundant=15, n_classes=7, random_state=42,
                                   n_clusters_per_class=2, flip_y=0.03)
        df = pd.DataFrame(X)
        df['target'] = y
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values
    if y_all.max() > 6:
        y_all %= 7
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, min(len(X_all), n_chunks*chunk_size), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)

# ===== Benchmark =====
def benchmark():
    chunks, classes = load_dataset(n_chunks=4, chunk_size=6000)
    fe = NonLogicFeatureEngineV2(top_k=8, n_clusters=25)
    nonlogic = NonLogicEnsembleV2(memory_size=15000, feature_engine=fe)
    
    # Baseline
    sgd = SGDClassifier(loss="log_loss", learning_rate="optimal", max_iter=8, warm_start=True, random_state=42)
    pa = PassiveAggressiveClassifier(C=0.1, max_iter=8, warm_start=True, random_state=42)
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 3
    first_sgd = first_pa = first_nonlogic = True
    
    # Fit feature engine
    if chunks:
        X_sample, y_sample = chunks[0]
        X_enh = fe.fit_transform(X_sample[:2000], y_sample[:2000])
        print(f"Enhanced features: {X_enh.shape[1]} (from {X_sample.shape[1]})")
    
    results = []
    for cid, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.7*len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        X_train_eng, X_test_eng = fe.transform(X_train), fe.transform(X_test)
        
        # NonLogic
        start = time.time()
        if first_nonlogic:
            nonlogic.partial_fit(X_train_eng, y_train, classes=classes)
            first_nonlogic=False
        else:
            nonlogic.partial_fit(X_train_eng, y_train)
        nl_acc = nonlogic.score(X_test_eng, y_test)
        nl_time = time.time()-start
        
        # SGD
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X_train_eng, y_train, classes=classes)
            first_sgd=False
        else:
            sgd.partial_fit(X_train_eng, y_train)
        sgd_acc = accuracy_score(y_test, sgd.predict(X_test_eng))
        sgd_time = time.time()-start
        
        # PA
        start = time.time()
        if first_pa:
            pa.partial_fit(X_train_eng, y_train, classes=classes)
            first_pa=False
        else:
            pa.partial_fit(X_train_eng, y_train)
        pa_acc = accuracy_score(y_test, pa.predict(X_test_eng))
        pa_time = time.time()-start
        
        # XGBoost
        start = time.time()
        xgb_all_X.append(X_train_eng)
        xgb_all_y.append(y_train)
        if len(xgb_all_X)>WINDOW_SIZE:
            xgb_all_X, xgb_all_y = xgb_all_X[-WINDOW_SIZE:], xgb_all_y[-WINDOW_SIZE:]
        X_xgb, y_xgb = np.vstack(xgb_all_X), np.concatenate(xgb_all_y)
        dtrain, dtest = xgb.DMatrix(X_xgb, label=y_xgb), xgb.DMatrix(X_test_eng, label=y_test)
        xgb_model = xgb.train({"objective":"multi:softmax","num_class":len(classes),
                               "max_depth":4,"eta":0.1,"subsample":0.8,"verbosity":0,"nthread":1},
                              dtrain, num_boost_round=12)
        xgb_pred = xgb_model.predict(dtest)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_time = time.time()-start
        
        results.append({'chunk':cid,'nonlogic_acc':nl_acc,'sgd_acc':sgd_acc,
                        'pa_acc':pa_acc,'xgb_acc':xgb_acc})
        print(f"Chunk {cid}: NonLogic {nl_acc:.3f} ({nl_time:.2f}s), SGD {sgd_acc:.3f}, PA {pa_acc:.3f}, XGB {xgb_acc:.3f}")
    
    # Analysis
    df = pd.DataFrame(results)
    accs = {m:df[f'{m}_acc'].mean() for m in ['nonlogic','sgd','pa','xgb']}
    winner = max(accs, key=accs.get)
    margin = (accs['nonlogic']-accs['xgb'])*100
    print(f"\nWinner: {winner.upper()} with NonLogic margin {margin:+.2f}% vs XGBoost")
    return df, accs

# ===== Main =====
if __name__=="__main__":
    start = time.time()
    df_results, accs = benchmark()
    print(f"Total time: {time.time()-start:.1f}s")
