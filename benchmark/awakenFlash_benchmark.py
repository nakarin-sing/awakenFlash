#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXED STREAMING BENCHMARK v12
Fixes:
1. Correct speed ratio calculation and interpretation
2. Better dataset handling (minimum batch requirements)
3. More meaningful final test evaluation
4. Clear separation of training time vs prediction quality
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"

import time
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

class StreamingEnsemble:
    def __init__(self, master_scaler, window_size=1500, update_interval=500):
        self.window_size = window_size
        self.update_interval = update_interval
        self.scaler = master_scaler
        
        self.sgd = SGDClassifier(
            loss='modified_huber', penalty='l2', alpha=0.005, 
            learning_rate='constant', eta0=0.0001, random_state=42, n_jobs=1 
        )
        
        self.rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, min_samples_split=20,
            max_samples=0.6, random_state=42, n_jobs=1
        )
        
        self.X_buffer = []
        self.y_buffer = []
        self.sample_count = 0
        self.is_fitted = False

    def _update_buffer(self, X_new, y_new):
        X_new_list = np.array(X_new).tolist()
        y_new_list = np.array(y_new).tolist()
        
        self.X_buffer.extend(X_new_list)
        self.y_buffer.extend(y_new_list)
        
        if len(self.X_buffer) > self.window_size:
            excess = len(self.X_buffer) - self.window_size
            self.X_buffer = self.X_buffer[excess:]
            self.y_buffer = self.y_buffer[excess:]
            
    def partial_fit(self, X_new, y_new):
        start_time = time.time()
        
        self._update_buffer(X_new, y_new)
        self.sample_count += len(X_new)
        
        X_window = np.array(self.X_buffer)
        y_window = np.array(self.y_buffer)
        
        update_needed = False
        if self.sample_count % self.update_interval == 0 or not self.is_fitted:
            update_needed = True

        X_scaled_new = self.scaler.transform(X_new)
        self.sgd.partial_fit(X_scaled_new, y_new, classes=np.unique(self.y_buffer))
        
        if update_needed:
            X_scaled_window = self.scaler.transform(X_window)
            self.rf.fit(X_scaled_window, y_window)
            
        self.is_fitted = True
        batch_time = time.time() - start_time
        return batch_time
    
    def predict(self, X):
        if not self.is_fitted or len(self.X_buffer) == 0:
            return np.zeros(len(X), dtype=int)
        
        try:
            X_scaled = self.scaler.transform(X)
        except:
            return np.zeros(len(X), dtype=int)
        
        if hasattr(self.sgd, 'predict_proba'):
            sgd_proba = self.sgd.predict_proba(X_scaled)
        else:
            dec_func = self.sgd.decision_function(X_scaled)
            sgd_proba = np.vstack([1 - dec_func, dec_func]).T
        
        if hasattr(self.rf, 'estimators_') and len(self.rf.estimators_) > 0:
            try:
                rf_proba = self.rf.predict_proba(X_scaled)
                if sgd_proba.shape == rf_proba.shape:
                    avg_proba = (rf_proba + sgd_proba) / 2
                    return np.argmax(avg_proba, axis=1)
                else:
                    return np.argmax(rf_proba, axis=1)
            except:
                return np.argmax(sgd_proba, axis=1)
        else:
            return np.argmax(sgd_proba, axis=1)

    def evaluate_on_val(self, X_val, y_val):
        if not self.is_fitted:
            return 0.0
        preds = self.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return acc

class StreamingXGBoost:
    def __init__(self, master_scaler, update_interval=500, window_size=1500):
        self.update_interval = update_interval
        self.window_size = window_size 
        self.scaler = master_scaler
        self.booster = None
        self.X_buffer = []
        self.y_buffer = []
        self.sample_count = 0
        self.is_fitted = False
        self.num_classes = 2
        
    def partial_fit(self, X_new, y_new):
        start_time = time.time()
        
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        
        self.X_buffer.append(X_new)
        self.y_buffer.append(y_new)
        self.sample_count += len(X_new)

        while np.sum([len(x) for x in self.X_buffer]) > self.window_size:
            self.X_buffer.pop(0)
            self.y_buffer.pop(0)

        X_all = np.vstack(self.X_buffer)
        y_all = np.hstack(self.y_buffer)
        
        X_scaled = self.scaler.transform(X_all)
        
        if self.sample_count % self.update_interval == 0 or self.booster is None:
            BOOST_ROUNDS = 20
            
            unique_classes = np.unique(y_all)
            self.num_classes = len(unique_classes)
            params = {
                'objective': 'multi:softprob' if self.num_classes > 2 else 'binary:logistic',
                'eval_metric': 'mlogloss',
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.1,
                'nthread': 1,
                'num_class': self.num_classes if self.num_classes > 2 else None
            }
            
            dtrain = xgb.DMatrix(X_scaled, label=y_all)
            
            self.booster = xgb.train(
                params, dtrain, num_boost_round=BOOST_ROUNDS, xgb_model=self.booster
            )
            self.is_fitted = True

        total_time = time.time() - start_time
        return total_time
    
    def predict(self, X):
        if not self.is_fitted or self.booster is None:
            return np.zeros(len(X), dtype=int)
        
        try:
            X_scaled = self.scaler.transform(X)
            dtest = xgb.DMatrix(X_scaled)
            
            if self.num_classes > 2:
                pred_proba = self.booster.predict(dtest)
                return np.argmax(pred_proba, axis=1)
            else:
                preds = self.booster.predict(dtest)
                return (preds > 0.5).astype(int)
        except Exception:
            return np.zeros(len(X), dtype=int)

    def evaluate_on_val(self, X_val, y_val):
        if not self.is_fitted:
            return 0.0
        preds = self.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return acc

def streaming_benchmark():
    print("🚀 FIXED STREAMING BENCHMARK v12")
    print("=" * 70)
    
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()), 
        ("Wine", load_wine())
    ]
    
    streaming_times = {
        'StreamingEnsemble': [],
        'XGBoostIncremental': []
    }
    
    final_test_accuracies = {
        'StreamingEnsemble': [],
        'XGBoostIncremental': []
    }
    
    for name, data in datasets:
        print(f"\n📊 Dataset: {name}")
        print("-" * 50)
        
        X, y = data.data, data.target
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
        )
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        
        master_scaler = StandardScaler()
        master_scaler.fit(X_train_full) 
        
        batch_size = 30  # Smaller batch for more updates
        n_batches = max(10, len(X_train) // batch_size)  # Ensure minimum batches
        print(f"Number of batches: {n_batches}")
        
        stream_ensemble = StreamingEnsemble(master_scaler=master_scaler, window_size=1500, update_interval=200)
        xgboost_stream = StreamingXGBoost(master_scaler=master_scaler, update_interval=200, window_size=1500)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))
            
            if start_idx >= len(X_train):
                break
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            t1 = stream_ensemble.partial_fit(X_batch, y_batch)
            t2 = xgboost_stream.partial_fit(X_batch, y_batch)
            
            if batch_idx > 0:
                streaming_times['StreamingEnsemble'].append(t1)
                streaming_times['XGBoostIncremental'].append(t2)
            
            if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
                acc1 = stream_ensemble.evaluate_on_val(X_val, y_val)
                acc2 = xgboost_stream.evaluate_on_val(X_val, y_val)
                
                print(f"Batch {batch_idx:3d} | "
                      f"StreamEns: {acc1:.4f}({t1*1000:.2f}ms) | "
                      f"XGBoostInc: {acc2:.4f}({t2*1000:.2f}ms)")

        final_acc_ens = accuracy_score(y_test, stream_ensemble.predict(X_test))
        final_acc_xgb = accuracy_score(y_test, xgboost_stream.predict(X_test))
        
        final_test_accuracies['StreamingEnsemble'].append(final_acc_ens)
        final_test_accuracies['XGBoostIncremental'].append(final_acc_xgb)
        
        print(f"\n✅ Final Test Accuracy: StreamEns={final_acc_ens:.4f}, XGBoost={final_acc_xgb:.4f}")
        
    print(f"\n{'='*70}")
    print("🏆 FINAL RESULTS")
    print(f"{'='*70}")
    
    ensemble_avg_time = np.mean(streaming_times['StreamingEnsemble']) * 1000  # ms
    xgb_avg_time = np.mean(streaming_times['XGBoostIncremental']) * 1000  # ms
    
    ensemble_final_acc = np.mean(final_test_accuracies['StreamingEnsemble'])
    xgb_final_acc = np.mean(final_test_accuracies['XGBoostIncremental'])
    
    print(f"\n📊 Average Streaming Update Time:")
    print(f"  Streaming Ensemble: {ensemble_avg_time:.2f}ms")
    print(f"  XGBoost Incremental: {xgb_avg_time:.2f}ms")
    
    print(f"\n🎯 Average Final Test Accuracy:")
    print(f"  Streaming Ensemble: {ensemble_final_acc:.4f}")
    print(f"  XGBoost Incremental: {xgb_final_acc:.4f}")
    
    # FIXED: Correct speed comparison
    if ensemble_avg_time < xgb_avg_time:
        speedup = xgb_avg_time / ensemble_avg_time
        print(f"\n⚡ Speed: Streaming Ensemble is {speedup:.2f}x FASTER")
    else:
        speedup = ensemble_avg_time / xgb_avg_time
        print(f"\n⚡ Speed: XGBoost is {speedup:.2f}x FASTER")
    
    acc_diff = abs(ensemble_final_acc - xgb_final_acc)
    if acc_diff < 0.01:
        print(f"🤝 Accuracy: Comparable (diff={acc_diff:.4f})")
    elif ensemble_final_acc > xgb_final_acc:
        print(f"📈 Accuracy: Streaming Ensemble wins by {acc_diff:.4f}")
    else:
        print(f"📈 Accuracy: XGBoost wins by {acc_diff:.4f}")


if __name__ == "__main__":
    streaming_benchmark()
