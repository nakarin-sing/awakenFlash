#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fair Streaming Benchmark: Online OneStep vs XGBoost
Goal: Beat XGBoost in streaming scenarios with FAIR comparison
MIT © 2025
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
import tracemalloc

# ========================================
# ONLINE ONESTEP WITH RECURSIVE UPDATES
# ========================================

class OnlineOneStep:
    """
    Online OneStep using Recursive Least Squares (RLS) algorithm
    - O(d²) updates with vectorized operations
    - Exponential forgetting to handle concept drift
    - Memory-efficient streaming learning
    - Polynomial features for non-linearity
    """
    def __init__(self, n_features, n_classes, forgetting_factor=0.99, reg_lambda=1e-3, 
                 use_poly=True, poly_degree=2):
        """
        Args:
            n_features: Number of input features
            n_classes: Number of output classes
            forgetting_factor: λ ∈ (0,1], closer to 1 = more memory
            reg_lambda: L2 regularization strength
            use_poly: Whether to use polynomial features
            poly_degree: Degree of polynomial features
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.forgetting_factor = forgetting_factor
        self.reg_lambda = reg_lambda
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        
        # Will be initialized after seeing first batch
        self.P = None
        self.W = None
        self.scaler = StandardScaler()
        self.poly = None
        self.is_fitted = False
        self.d_total = None
        
    def _partial_fit_rls(self, X, y):
        """
        Mini-batch RLS with Sherman-Morrison updates
        Memory efficient: O(d²) storage, O(batch × d²) computation
        """
        X = X.astype(np.float32)
        n_samples = X.shape[0]
        
        # One-hot encode targets
        y_onehot = np.zeros((n_samples, self.n_classes), dtype=np.float32)
        y_onehot[np.arange(n_samples), y] = 1.0
        
        # Apply exponential forgetting: P = P / λ
        self.P = self.P / self.forgetting_factor
        
        # Process in smaller mini-batches to avoid memory explosion
        mini_batch_size = min(100, n_samples)
        
        for start_idx in range(0, n_samples, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            y_batch = y_onehot[start_idx:end_idx]
            
            # Sherman-Morrison updates for mini-batch
            for i in range(X_batch.shape[0]):
                x = X_batch[i:i+1].T  # (d, 1)
                y_i = y_batch[i:i+1].T  # (c, 1)
                
                # Kalman gain: K = P x / (λ + x^T P x)
                Px = self.P @ x  # (d, 1)
                denom = self.forgetting_factor + (x.T @ Px)[0, 0]
                
                # Numerical stability check
                if denom < 1e-10:
                    continue
                
                K = Px / denom  # (d, 1)
                
                # Prediction error
                pred_error = y_i.T - (x.T @ self.W)  # (1, c)
                
                # Update weights: W = W + K * error^T
                self.W = self.W + K @ pred_error  # (d, c)
                
                # Update P: P = P - K K^T * (λ + x^T P x)
                self.P = self.P - (K @ K.T) * denom  # (d, d)
    
    def partial_fit(self, X, y):
        """
        Update model with new chunk of data
        """
        # Fit or update scaler
        if not self.is_fitted:
            X = self.scaler.fit_transform(X)
            
            # Initialize polynomial features if needed
            if self.use_poly:
                from sklearn.preprocessing import PolynomialFeatures
                self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
                X_features = self.poly.fit_transform(X).astype(np.float32)
            else:
                X_features = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
            
            # Initialize P and W based on actual feature size
            self.d_total = X_features.shape[1]
            self.P = np.eye(self.d_total, dtype=np.float32) / self.reg_lambda
            self.W = np.zeros((self.d_total, self.n_classes), dtype=np.float32)
            self.is_fitted = True
        else:
            X = self.scaler.transform(X)
            if self.use_poly:
                X_features = self.poly.transform(X).astype(np.float32)
            else:
                X_features = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        
        # Update weights using RLS
        self._partial_fit_rls(X_features, y)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        X = self.scaler.transform(X)
        if self.use_poly:
            X_features = self.poly.transform(X).astype(np.float32)
        else:
            X_features = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        
        # Compute logits
        logits = X_features @ self.W
        
        # Softmax for probabilities
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        """
        Predict class labels
        """
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)

# ========================================
# BATCH ONESTEP (FOR COMPARISON)
# ========================================

class BatchOneStep:
    """
    Batch OneStep that accumulates all data before training
    """
    def __init__(self, reg_lambda=1e-3):
        self.reg_lambda = reg_lambda
        self.scaler = StandardScaler()
        self.W = None
        self.X_accumulated = []
        self.y_accumulated = []
        
    def partial_fit(self, X, y):
        """Accumulate data"""
        self.X_accumulated.append(X)
        self.y_accumulated.append(y)
        return self
    
    def fit_accumulated(self):
        """Train on all accumulated data"""
        X_all = np.vstack(self.X_accumulated)
        y_all = np.hstack(self.y_accumulated)
        
        # Preprocess
        X_all = self.scaler.fit_transform(X_all)
        X_all = np.hstack([np.ones((X_all.shape[0], 1), dtype=np.float32), X_all])
        
        # One-hot encode
        n_classes = y_all.max() + 1
        y_onehot = np.zeros((len(y_all), n_classes), dtype=np.float32)
        y_onehot[np.arange(len(y_all)), y_all] = 1.0
        
        # Solve closed-form
        XTX = X_all.T @ X_all
        XTY = X_all.T @ y_onehot
        lambda_adaptive = self.reg_lambda * np.trace(XTX) / XTX.shape[0]
        I = np.eye(XTX.shape[0], dtype=np.float32)
        
        self.W = np.linalg.solve(XTX + lambda_adaptive * I, XTY)
        
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        X = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        logits = X @ self.W
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

# ========================================
# FAIR STREAMING BENCHMARK
# ========================================

def generate_streaming_data(n_chunks=12, samples_per_chunk=10000, n_features=20, 
                           n_classes=2, concept_drift=True, random_state=42):
    """
    Generate synthetic streaming data with optional concept drift
    """
    np.random.seed(random_state)
    chunks = []
    
    for i in range(n_chunks):
        # Simulate concept drift by changing cluster centers
        if concept_drift and i > 0:
            drift_factor = i * 0.1
            cluster_std = 1.0 + drift_factor * 0.5
            flip_y = 0.05 * drift_factor  # Add label noise
        else:
            cluster_std = 1.0
            flip_y = 0.0
        
        X, y = make_classification(
            n_samples=samples_per_chunk,
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            n_redundant=int(n_features * 0.1),
            n_classes=n_classes,
            n_clusters_per_class=2,
            weights=None,
            flip_y=flip_y,
            class_sep=1.0,
            hypercube=True,
            shift=0.0,
            scale=1.0,
            shuffle=True,
            random_state=random_state + i
        )
        
        # Simulate concept drift by adding noise
        if concept_drift and i > 0:
            drift_factor = i * 0.1
            noise = np.random.normal(0, cluster_std - 1.0, X.shape)
            X = X + noise
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state + i
        )
        
        chunks.append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'chunk_id': i + 1
        })
    
    return chunks

def benchmark_streaming():
    """
    Fair benchmark: Online OneStep vs XGBoost vs Batch OneStep
    """
    print("=" * 80)
    print("FAIR STREAMING BENCHMARK")
    print("=" * 80)
    print("Generating streaming data with concept drift...")
    
    # Generate data
    chunks = generate_streaming_data(
        n_chunks=12,
        samples_per_chunk=5000,  # Smaller for faster testing
        n_features=20,
        n_classes=2,
        concept_drift=True,
        random_state=42
    )
    
    n_features = chunks[0]['X_train'].shape[1]
    n_classes = 2
    
    print(f"Dataset: 12 chunks × 5000 samples × {n_features} features")
    print(f"Classes: {n_classes} (binary classification)")
    print("=" * 80)
    
    # Initialize models
    online_onestep = OnlineOneStep(
        n_features=n_features,
        n_classes=n_classes,
        forgetting_factor=0.995,  # Less aggressive forgetting
        reg_lambda=0.1,  # Stronger regularization for stability
        use_poly=False,  # Disable poly for now - too many features
        poly_degree=2
    )
    
    batch_onestep = BatchOneStep(reg_lambda=1e-2)
    
    # XGBoost will be retrained on all accumulated data (fair comparison)
    xgb_X_accumulated = []
    xgb_y_accumulated = []
    
    # Results storage
    results = {
        'chunk': [],
        'online_auc': [],
        'online_acc': [],
        'online_time': [],
        'online_memory': [],
        'batch_auc': [],
        'batch_acc': [],
        'batch_time': [],
        'batch_memory': [],
        'xgb_auc': [],
        'xgb_acc': [],
        'xgb_time': [],
        'xgb_memory': []
    }
    
    # Process each chunk
    for chunk_data in chunks:
        chunk_id = chunk_data['chunk_id']
        X_train = chunk_data['X_train']
        X_test = chunk_data['X_test']
        y_train = chunk_data['y_train']
        y_test = chunk_data['y_test']
        
        print(f"\n{'='*80}")
        print(f"Processing Chunk {chunk_id}/12")
        print(f"{'='*80}")
        
        # =====================================================================
        # 1. ONLINE ONESTEP (Incremental Update)
        # =====================================================================
        tracemalloc.start()
        t0 = time.time()
        
        online_onestep.partial_fit(X_train, y_train)
        y_pred_online = online_onestep.predict(X_test)
        y_proba_online = online_onestep.predict_proba(X_test)[:, 1]
        
        online_time = time.time() - t0
        _, online_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        online_auc = roc_auc_score(y_test, y_proba_online)
        online_acc = accuracy_score(y_test, y_pred_online)
        
        print(f"[Online OneStep] AUC: {online_auc:.4f}, ACC: {online_acc:.4f}, "
              f"Time: {online_time:.4f}s, Memory: {online_peak_mem/1024/1024:.2f}MB")
        
        # =====================================================================
        # 2. BATCH ONESTEP (Retrain on All Data)
        # =====================================================================
        tracemalloc.start()
        t0 = time.time()
        
        batch_onestep.partial_fit(X_train, y_train)
        batch_onestep.fit_accumulated()
        y_pred_batch = batch_onestep.predict(X_test)
        y_proba_batch = batch_onestep.predict_proba(X_test)[:, 1]
        
        batch_time = time.time() - t0
        _, batch_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        batch_auc = roc_auc_score(y_test, y_proba_batch)
        batch_acc = accuracy_score(y_test, y_pred_batch)
        
        print(f"[Batch OneStep]  AUC: {batch_auc:.4f}, ACC: {batch_acc:.4f}, "
              f"Time: {batch_time:.4f}s, Memory: {batch_peak_mem/1024/1024:.2f}MB")
        
        # =====================================================================
        # 3. XGBOOST (Retrain on All Data - Fair!)
        # =====================================================================
        xgb_X_accumulated.append(X_train)
        xgb_y_accumulated.append(y_train)
        
        tracemalloc.start()
        t0 = time.time()
        
        X_xgb_all = np.vstack(xgb_X_accumulated)
        y_xgb_all = np.hstack(xgb_y_accumulated)
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        xgb_model.fit(X_xgb_all, y_xgb_all)
        
        y_pred_xgb = xgb_model.predict(X_test)
        y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        xgb_time = time.time() - t0
        _, xgb_peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        xgb_auc = roc_auc_score(y_test, y_proba_xgb)
        xgb_acc = accuracy_score(y_test, y_pred_xgb)
        
        print(f"[XGBoost]        AUC: {xgb_auc:.4f}, ACC: {xgb_acc:.4f}, "
              f"Time: {xgb_time:.4f}s, Memory: {xgb_peak_mem/1024/1024:.2f}MB")
        
        # Store results
        results['chunk'].append(chunk_id)
        results['online_auc'].append(online_auc)
        results['online_acc'].append(online_acc)
        results['online_time'].append(online_time)
        results['online_memory'].append(online_peak_mem / 1024 / 1024)
        results['batch_auc'].append(batch_auc)
        results['batch_acc'].append(batch_acc)
        results['batch_time'].append(batch_time)
        results['batch_memory'].append(batch_peak_mem / 1024 / 1024)
        results['xgb_auc'].append(xgb_auc)
        results['xgb_acc'].append(xgb_acc)
        results['xgb_time'].append(xgb_time)
        results['xgb_memory'].append(xgb_peak_mem / 1024 / 1024)
        
        # Quick comparison
        print(f"\n{'─'*80}")
        print(f"Chunk {chunk_id} Winners:")
        print(f"  AUC:    {'Online' if online_auc > max(batch_auc, xgb_auc) else 'Batch' if batch_auc > xgb_auc else 'XGBoost'}")
        print(f"  Speed:  {'Online' if online_time < min(batch_time, xgb_time) else 'Batch' if batch_time < xgb_time else 'XGBoost'}")
        print(f"  Memory: {'Online' if online_peak_mem < min(batch_peak_mem, xgb_peak_mem) else 'Batch' if batch_peak_mem < xgb_peak_mem else 'XGBoost'}")
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    df = pd.DataFrame(results)
    
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - AVERAGE ACROSS ALL CHUNKS")
    print(f"{'='*80}\n")
    
    print(f"{'Metric':<20} {'Online OneStep':<18} {'Batch OneStep':<18} {'XGBoost':<18} {'Winner':<15}")
    print(f"{'─'*90}")
    
    # AUC
    online_auc_avg = df['online_auc'].mean()
    batch_auc_avg = df['batch_auc'].mean()
    xgb_auc_avg = df['xgb_auc'].mean()
    auc_winner = 'Online' if online_auc_avg >= max(batch_auc_avg, xgb_auc_avg) else 'Batch' if batch_auc_avg >= xgb_auc_avg else 'XGBoost'
    print(f"{'AUC':<20} {online_auc_avg:<18.4f} {batch_auc_avg:<18.4f} {xgb_auc_avg:<18.4f} {auc_winner:<15}")
    
    # Accuracy
    online_acc_avg = df['online_acc'].mean()
    batch_acc_avg = df['batch_acc'].mean()
    xgb_acc_avg = df['xgb_acc'].mean()
    acc_winner = 'Online' if online_acc_avg >= max(batch_acc_avg, xgb_acc_avg) else 'Batch' if batch_acc_avg >= xgb_acc_avg else 'XGBoost'
    print(f"{'Accuracy':<20} {online_acc_avg:<18.4f} {batch_acc_avg:<18.4f} {xgb_acc_avg:<18.4f} {acc_winner:<15}")
    
    # Total Time
    online_time_total = df['online_time'].sum()
    batch_time_total = df['batch_time'].sum()
    xgb_time_total = df['xgb_time'].sum()
    time_winner = 'Online' if online_time_total <= min(batch_time_total, xgb_time_total) else 'Batch' if batch_time_total <= xgb_time_total else 'XGBoost'
    print(f"{'Total Time (s)':<20} {online_time_total:<18.4f} {batch_time_total:<18.4f} {xgb_time_total:<18.4f} {time_winner:<15}")
    
    # Average Memory
    online_mem_avg = df['online_memory'].mean()
    batch_mem_avg = df['batch_memory'].mean()
    xgb_mem_avg = df['xgb_memory'].mean()
    mem_winner = 'Online' if online_mem_avg <= min(batch_mem_avg, xgb_mem_avg) else 'Batch' if batch_mem_avg <= xgb_mem_avg else 'XGBoost'
    print(f"{'Avg Memory (MB)':<20} {online_mem_avg:<18.2f} {batch_mem_avg:<18.2f} {xgb_mem_avg:<18.2f} {mem_winner:<15}")
    
    print(f"\n{'='*80}")
    
    # Overall Winner
    online_wins = sum([
        online_auc_avg >= max(batch_auc_avg, xgb_auc_avg),
        online_acc_avg >= max(batch_acc_avg, xgb_acc_avg),
        online_time_total <= min(batch_time_total, xgb_time_total),
        online_mem_avg <= min(batch_mem_avg, xgb_mem_avg)
    ])
    
    batch_wins = sum([
        batch_auc_avg >= max(online_auc_avg, xgb_auc_avg),
        batch_acc_avg >= max(online_acc_avg, xgb_acc_avg),
        batch_time_total <= min(online_time_total, xgb_time_total),
        batch_mem_avg <= min(online_mem_avg, xgb_mem_avg)
    ])
    
    xgb_wins = sum([
        xgb_auc_avg >= max(online_auc_avg, batch_auc_avg),
        xgb_acc_avg >= max(online_acc_avg, batch_acc_avg),
        xgb_time_total <= min(online_time_total, batch_time_total),
        xgb_mem_avg <= min(online_mem_avg, batch_mem_avg)
    ])
    
    print(f"OVERALL WINNER:")
    print(f"  Online OneStep: {online_wins}/4 metrics")
    print(f"  Batch OneStep:  {batch_wins}/4 metrics")
    print(f"  XGBoost:        {xgb_wins}/4 metrics")
    
    if online_wins >= max(batch_wins, xgb_wins):
        print(f"\n🏆 ONLINE ONESTEP WINS!")
        print("   ✓ Incremental learning with O(d²) updates")
        print("   ✓ Handles concept drift with forgetting factor")
        print("   ✓ Fast and memory efficient")
    elif batch_wins >= xgb_wins:
        print(f"\n🏆 BATCH ONESTEP WINS!")
        print("   ✓ Optimal closed-form solution")
        print("   ✓ Better than incremental when no drift")
    else:
        print(f"\n🏆 XGBOOST WINS!")
        print("   ✓ Tree ensemble handles complex patterns")
    
    print(f"{'='*80}")
    
    # Save results
    df.to_csv('streaming_results.csv', index=False)
    print(f"\n💾 Results saved to streaming_results.csv")

if __name__ == "__main__":
    benchmark_streaming()
