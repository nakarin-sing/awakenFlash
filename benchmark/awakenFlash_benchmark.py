#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEBUG VERSION v14 - Diagnose why SGD is not learning
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
    def __init__(self, master_scaler, all_classes, window_size=1500, update_interval=500):
        self.window_size = window_size
        self.update_interval = update_interval
        self.scaler = master_scaler
        self.all_classes = np.array(all_classes)
        
        # DIAGNOSTIC: Try different SGD configuration
        self.sgd = SGDClassifier(
            loss='log_loss',  # Changed from modified_huber
            penalty='l2', 
            alpha=0.0001,  # Reduced regularization
            learning_rate='optimal',  # Changed from constant
            random_state=42, 
            n_jobs=1,
            max_iter=1000,
            tol=1e-3
        )
        
        self.rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, min_samples_split=20,
            max_samples=0.6, random_state=42, n_jobs=1
        )
        
        self.X_buffer = []
        self.y_buffer = []
        self.sample_count = 0
        self.is_fitted = False
        self.sgd_classes_seen = set()  # DIAGNOSTIC

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

        # SGD update with all classes
        X_scaled_new = self.scaler.transform(X_new)
        
        # DIAGNOSTIC: Track which classes SGD has seen
        self.sgd_classes_seen.update(y_new)
        
        self.sgd.partial_fit(X_scaled_new, y_new, classes=self.all_classes)
        
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
        
        # DIAGNOSTIC: Try using only SGD first
        try:
            sgd_pred = self.sgd.predict(X_scaled)
            
            # If RF is fitted, ensemble
            if hasattr(self.rf, 'estimators_') and len(self.rf.estimators_) > 0:
                try:
                    rf_pred = self.rf.predict(X_scaled)
                    # Simple voting
                    combined = np.vstack([sgd_pred, rf_pred])
                    final_pred = np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(), 
                        axis=0, 
                        arr=combined
                    )
                    return final_pred
                except:
                    return sgd_pred
            else:
                return sgd_pred
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.zeros(len(X), dtype=int)

    def evaluate_on_val(self, X_val, y_val):
        if not self.is_fitted:
            return 0.0
        preds = self.predict(X_val)
        acc = accuracy_score(y_val, preds)
        return acc
    
    def get_diagnostics(self):
        """Return diagnostic information"""
        return {
            'sgd_classes_seen': self.sgd_classes_seen,
            'sgd_classes': getattr(self.sgd, 'classes_', None),
            'buffer_size': len(self.X_buffer),
            'sample_count': self.sample_count
        }

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
    print("🔍 DEBUG VERSION v14 - Diagnosing SGD Issue")
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
        
        all_classes = np.unique(y)
        print(f"All Classes: {all_classes}")
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
        )
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        print(f"Train classes distribution: {np.bincount(y_train)}")
        
        master_scaler = StandardScaler()
        master_scaler.fit(X_train_full) 
        
        batch_size = 30
        n_batches = max(10, len(X_train) // batch_size)
        
        stream_ensemble = StreamingEnsemble(
            master_scaler=master_scaler, 
            all_classes=all_classes,
            window_size=1500, 
            update_interval=200
        )
        xgboost_stream = StreamingXGBoost(
            master_scaler=master_scaler, 
            update_interval=200, 
            window_size=1500
        )
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))
            
            if start_idx >= len(X_train):
                break
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            # DIAGNOSTIC: Print first batch classes
            if batch_idx == 0:
                print(f"First batch classes: {np.unique(y_batch)}")
            
            t1 = stream_ensemble.partial_fit(X_batch, y_batch)
            t2 = xgboost_stream.partial_fit(X_batch, y_batch)
            
            if batch_idx > 0:
                streaming_times['StreamingEnsemble'].append(t1)
                streaming_times['XGBoostIncremental'].append(t2)
            
            if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
                acc1 = stream_ensemble.evaluate_on_val(X_val, y_val)
                acc2 = xgboost_stream.evaluate_on_val(X_val, y_val)
                
                # DIAGNOSTIC: Show what SGD has learned
                diag = stream_ensemble.get_diagnostics()
                
                print(f"Batch {batch_idx:3d} | "
                      f"StreamEns: {acc1:.4f}({t1*1000:.2f}ms) | "
                      f"XGBoostInc: {acc2:.4f}({t2*1000:.2f}ms)")
                print(f"  → SGD saw classes: {sorted(diag['sgd_classes_seen'])}, "
                      f"SGD trained on: {diag['sgd_classes']}")

        final_acc_ens = accuracy_score(y_test, stream_ensemble.predict(X_test))
        final_acc_xgb = accuracy_score(y_test, xgboost_stream.predict(X_test))
        
        print(f"\n✅ Final Test Accuracy: StreamEns={final_acc_ens:.4f}, XGBoost={final_acc_xgb:.4f}")
        
        # DIAGNOSTIC: Final model state
        print(f"📋 Final Diagnostics: {stream_ensemble.get_diagnostics()}")
        
        final_test_accuracies['StreamingEnsemble'].append(final_acc_ens)
        final_test_accuracies['XGBoostIncremental'].append(final_acc_xgb)
        
    print(f"\n{'='*70}")
    print("🏆 SUMMARY")
    print(f"{'='*70}")
    
    ensemble_avg_time = np.mean(streaming_times['StreamingEnsemble']) * 1000
    xgb_avg_time = np.mean(streaming_times['XGBoostIncremental']) * 1000
    
    ensemble_final_acc = np.mean(final_test_accuracies['StreamingEnsemble'])
    xgb_final_acc = np.mean(final_test_accuracies['XGBoostIncremental'])
    
    print(f"\nAvg Update Time: StreamEns={ensemble_avg_time:.2f}ms, XGBoost={xgb_avg_time:.2f}ms")
    print(f"Avg Test Acc: StreamEns={ensemble_final_acc:.4f}, XGBoost={xgb_final_acc:.4f}")
    
    if ensemble_avg_time < xgb_avg_time:
        speedup = xgb_avg_time / ensemble_avg_time
        print(f"\n⚡ Streaming Ensemble is {speedup:.2f}x FASTER")
    else:
        speedup = ensemble_avg_time / xgb_avg_time
        print(f"\n⚡ XGBoost is {speedup:.2f}x FASTER")


if __name__ == "__main__":
    streaming_benchmark()
