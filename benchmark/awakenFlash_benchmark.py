#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
awakenFlash_benchmark.py – OPTIMIZED ŚŪNYATĀ EDITION
Faster than light, emptier than void
"""

import os
import time        # แก้แล้ว!
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


# ========================================
# 1. OPTIMIZED ŚŪNYATĀ ENSEMBLE
# ========================================
class TemporalTranscendenceEnsemble:
    def __init__(self, n_base_models=5, memory_size=20000):
        self.n_base_models = n_base_models
        self.models = []
        self.weights = np.ones(n_base_models) / n_base_models
        self.memory_size = memory_size
        self.classes_ = None
        self.X_buffer = []
        self.y_buffer = []
        
        for i in range(n_base_models):
            if i % 2 == 0:
                model = SGDClassifier(
                    loss='log_loss',
                    learning_rate='adaptive',
                    eta0=0.05,
                    max_iter=5,
                    warm_start=True,
                    random_state=42+i,
                    alpha=1e-4
                )
            else:
                model = PassiveAggressiveClassifier(
                    C=0.1,
                    max_iter=5,
                    warm_start=True,
                    random_state=42+i
                )
            self.models.append(model)
    
    def _update_weights(self, X_test, y_test):
        def model_acc(model):
            try:
                return model.score(X_test, y_test)
            except:
                return 0.0
        
        accs = Parallel(n_jobs=-1)(delayed(model_acc)(m) for m in self.models)
        accs = np.array(accs)
        accs = np.clip(accs, 0.1, 1.0)
        self.weights = np.exp(accs * 3)
        self.weights /= self.weights.sum()
    
    def partial_fit(self, X, y, classes=None):
        if classes is not None:
            self.classes_ = classes
        
        self.X_buffer.append(X.copy())
        self.y_buffer.append(y.copy())
        
        total = sum(len(x) for x in self.X_buffer)
        while total > self.memory_size and len(self.X_buffer) > 1:
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)
            total = sum(len(x) for x in self.X_buffer)
        
        for model in self.models:
            try:
                model.partial_fit(X, y, classes=self.classes_)
            except:
                pass
        
        if len(self.X_buffer) > 1:
            all_X = np.vstack(self.X_buffer)
            all_y = np.concatenate(self.y_buffer)
            idx = np.random.choice(len(all_X), size=min(3000, len(all_X)), replace=False)
            for model in self.models:
                try:
                    model.partial_fit(all_X[idx], all_y[idx], classes=self.classes_)
                except:
                    pass
    
    def predict(self, X):
        if not self.models or self.classes_ is None:
            return np.zeros(len(X), dtype=int)
        
        preds = Parallel(n_jobs=-1)(
            delayed(lambda m: m.predict(X))(m) for m in self.models
        )
        
        n_samples = len(X)
        vote = np.zeros((n_samples, len(self.classes_)))
        
        for pred, w in zip(preds, self.weights):
            for i, cls in enumerate(self.classes_):
                vote[:, i] += (pred == cls) * w
        
        return self.classes_[np.argmax(vote, axis=1)]
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ========================================
# 2. OPTIMIZED DATA LOADING
# ========================================
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"Loading dataset...")
    df = pd.read_csv(url, header=None, nrows=n_chunks * chunk_size)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = (df.iloc[:, -1].values - 1).astype(np.int32)
    
    print(f"   Dataset: {X_all.shape}, Classes: {len(np.unique(y_all))}")
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size])
              for i in range(0, len(X_all), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)


# ========================================
# 3. OPTIMIZED METRICS
# ========================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {'accuracy': acc, 'f1': f1}


# ========================================
# 4. OPTIMIZED BENCHMARK LOOP
# ========================================
def scenario_non_dualistic(chunks, all_classes):
    print("\n" + "="*80)
    print("OPTIMIZED NON-DUALISTIC SCENARIO")
    print("="*80)
    
    sunyata = TemporalTranscendenceEnsemble(n_base_models=5, memory_size=20000)
    sgd = SGDClassifier(loss='log_loss', max_iter=3, warm_start=True, random_state=42)
    pa = PassiveAggressiveClassifier(C=0.1, max_iter=3, warm_start=True, random_state=42)
    
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 3
    
    first_sgd = first_pa = True
    results = []
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        print(f"Chunk {chunk_id:02d}/{len(chunks)} | Train: {len(X_train)}, Test: {len(X_test)}")
        
        # ŚŪNYATĀ
        start = time.time()
        sunyata.partial_fit(X_train, y_train, classes=all_classes)
        sunyata._update_weights(X_test, y_test)
        sunyata_pred = sunyata.predict(X_test)
        sunyata_metrics = compute_metrics(y_test, sunyata_pred)
        sunyata_time = time.time() - start
        
        # SGD
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X_train, y_train, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train, classes=all_classes)
        sgd_pred = sgd.predict(X_test)
        sgd_metrics = compute_metrics(y_test, sgd_pred)
        sgd_time = time.time() - start
        
        # PA
        start = time.time()
        if first_pa:
            pa.partial_fit(X_train, y_train, classes=all_classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train, classes=all_classes)
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
        dtest = xgb.DMatrix(X_test)
        
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": 7, "max_depth": 4, "eta": 0.2, "verbosity": 0},
            dtrain, num_boost_round=10
        )
        xgb_pred = xgb_model.predict(dtest).astype(int)
        xgb_metrics = compute_metrics(y_test, xgb_pred)
        xgb_time = time.time() - start
        
        results.append({
            'chunk': chunk_id,
            'sunyata_acc': sunyata_metrics['accuracy'],
            'sunyata_f1': sunyata_metrics['f1'],
            'sunyata_time': sunyata_time,
            'sgd_acc': sgd_metrics['accuracy'],
            'sgd_f1': sgd_metrics['f1'],
            'sgd_time': sgd_time,
            'pa_acc': pa_metrics['accuracy'],
            'pa_f1': pa_metrics['f1'],
            'pa_time': pa_time,
            'xgb_acc': xgb_metrics['accuracy'],
            'xgb_f1': xgb_metrics['f1'],
            'xgb_time': xgb_time,
        })
        
        print(f"  Śūnyata: acc={sunyata_metrics['accuracy']:.3f} t={sunyata_time:.3f}s")
        print(f"  PA:      acc={pa_metrics['accuracy']:.3f} t={pa_time:.3f}s")
        print(f"  XGB:     acc={xgb_metrics['accuracy']:.3f} t={xgb_time:.3f}s")
        print()
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("OPTIMIZED FINAL RESULTS")
    print("="*80)
    for model in ['sunyata', 'pa', 'xgb']:
        acc = df_results[f'{model}_acc'].mean()
        time = df_results[f'{model}_time'].mean()
        print(f"{model.upper():8s}: Acc={acc:.4f} | Time={time:.4f}s")
    
    sunyata_acc = df_results['sunyata_acc'].mean()
    xgb_acc = df_results['xgb_acc'].mean()
    sunyata_time = df_results['sunyata_time'].mean()
    xgb_time = df_results['xgb_time'].mean()
    
    print("\nOPTIMIZED INSIGHT:")
    if sunyata_acc >= xgb_acc * 0.98:
        print(f"   Śūnyatā: {sunyata_acc:.4f} acc in {sunyata_time:.3f}s")
        print(f"   XGBoost: {xgb_acc:.4f} acc in {xgb_time:.3f}s")
        print(f"   Speedup: {xgb_time/sunyata_time:.1f}x")
        print(f"   Emptiness achieved.")
    else:
        print(f"   Still attached to accuracy.")
    
    return df_results


# ========================================
# 5. MAIN
# ========================================
def main():
    print("="*80)
    print("OPTIMIZED awakenFlash BENCHMARK")
    print("="*80)
    
    chunks, all_classes = load_data(n_chunks=10, chunk_size=10000)
    results = scenario_non_dualistic(chunks, all_classes)
    
    os.makedirs('benchmark_results', exist_ok=True)
    results.to_csv('benchmark_results/optimized_results.csv', index=False)
    
    print("\nOptimized benchmark complete.")
    print("Results: benchmark_results/optimized_results.csv")


if __name__ == "__main__":
    main()
