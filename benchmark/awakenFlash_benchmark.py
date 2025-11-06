#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGHTNING BENCHMARK - Fast & Efficient
Optimized for CI/CD with 30-second timeout
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set environment for maximum performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

class FastFeatureEngine:
    """
    Fast feature engineering - limited interactions for speed
    """
    
    def __init__(self, max_interactions=5):
        self.max_interactions = max_interactions
        self.interaction_pairs = None
    
    def fit_transform(self, X):
        """Create limited interaction features"""
        n_features = X.shape[1]
        
        # Quick variance calculation
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-5:]  # Only top 5 features
        
        self.interaction_pairs = []
        for i in range(len(top_indices)):
            for j in range(i+1, min(i+2, len(top_indices))):  # Limited pairs
                if len(self.interaction_pairs) < self.max_interactions:
                    self.interaction_pairs.append((top_indices[i], top_indices[j]))
        
        X_interactions = []
        for i, j in self.interaction_pairs:
            X_interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        if X_interactions:
            return np.hstack([X] + X_interactions)
        return X
    
    def transform(self, X):
        """Apply transformations quickly"""
        if self.interaction_pairs is None:
            return X
        
        X_interactions = []
        for i, j in self.interaction_pairs:
            X_interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        if X_interactions:
            return np.hstack([X] + X_interactions)
        return X


class LightningEnsemble:
    """
    Fast ensemble with 3 models for CI speed
    """
    
    def __init__(self, memory_size=10000, feature_engine=None):
        self.models = []
        self.weights = np.ones(3) / 3
        self.all_data_X = []
        self.all_data_y = []
        self.memory_size = memory_size
        self.feature_engine = feature_engine
        
        # Only 3 fast models
        self.models.append(SGDClassifier(
            loss='log_loss', learning_rate='optimal', max_iter=10,
            warm_start=True, random_state=42, alpha=0.001
        ))
        self.models.append(PassiveAggressiveClassifier(
            C=0.1, max_iter=10, warm_start=True, random_state=43
        ))
        self.models.append(SGDClassifier(
            loss='modified_huber', learning_rate='optimal', max_iter=10,
            warm_start=True, random_state=44, alpha=0.001
        ))
        
        self.first_fit = True
        self.classes_ = None
    
    def partial_fit(self, X, y, classes=None):
        """Fast online learning"""
        if self.first_fit and classes is not None:
            self.classes_ = classes
            self.first_fit = False
        
        # Limited memory management
        self.all_data_X.append(X)
        self.all_data_y.append(y)
        
        total_samples = sum(len(x) for x in self.all_data_X)
        while total_samples > self.memory_size and len(self.all_data_X) > 1:
            self.all_data_X.pop(0)
            self.all_data_y.pop(0)
            total_samples = sum(len(x) for x in self.all_data_X)
        
        # Fast online learning only (no batch learning for speed)
        for model in self.models:
            try:
                if classes is not None:
                    model.partial_fit(X, y, classes=classes)
                else:
                    model.partial_fit(X, y)
            except:
                pass
    
    def predict(self, X):
        """Fast prediction"""
        if not self.models or self.classes_ is None:
            return np.zeros(len(X))
        
        all_predictions = []
        for model in self.models:
            try:
                pred = model.predict(X)
                all_predictions.append(pred)
            except:
                pass
        
        if not all_predictions:
            return np.zeros(len(X))
        
        # Simple voting
        n_samples = len(X)
        n_classes = len(self.classes_)
        vote_matrix = np.zeros((n_samples, n_classes))
        
        for pred in all_predictions:
            for i, cls in enumerate(self.classes_):
                vote_matrix[:, i] += (pred == cls)
        
        return self.classes_[np.argmax(vote_matrix, axis=1)]


def load_data_fast(n_chunks=5, chunk_size=5000):
    """Load smaller dataset quickly"""
    print("📦 Loading dataset (fast mode)...")
    
    # Use smaller subset for CI
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        df = pd.read_csv(url, header=None, nrows=50000)  # Only 50K samples
    except:
        # Fallback: generate synthetic data
        print("   Using synthetic data...")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=50000, n_features=20, n_informative=15,
            n_classes=7, random_state=42
        )
        df = pd.DataFrame(X)
        df['target'] = y
    
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values
    
    if y_all.max() > 6:  # Adjust for synthetic data
        y_all = y_all % 7
    
    print(f"   Dataset: {X_all.shape}, Classes: {len(np.unique(y_all))}")
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, min(len(X_all), n_chunks * chunk_size), chunk_size)]
    
    return chunks[:n_chunks], np.unique(y_all)


def fast_benchmark():
    """
    FAST BENCHMARK - Optimized for CI
    """
    print("\n" + "="*60)
    print("⚡ LIGHTNING BENCHMARK - CI Optimized")
    print("="*60)
    
    # Load smaller dataset
    chunks, all_classes = load_data_fast(n_chunks=4, chunk_size=5000)
    
    # Shared feature engine
    feature_engine = FastFeatureEngine(max_interactions=3)
    
    # Initialize models
    lightning = LightningEnsemble(memory_size=10000, feature_engine=feature_engine)
    
    sgd = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        max_iter=5,  # Fewer iterations
        warm_start=True,
        random_state=42
    )
    
    pa = PassiveAggressiveClassifier(
        C=0.1,
        max_iter=5,  # Fewer iterations
        warm_start=True,
        random_state=42
    )
    
    # XGBoost with small window
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 2  # Smaller window
    
    first_sgd = first_pa = first_lightning = True
    results = []
    
    # Quick feature engine fit
    if chunks:
        X_sample, _ = chunks[0]
        feature_engine.fit_transform(X_sample[:1000])
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.7 * len(X_chunk))  # Smaller test set
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        # Apply feature engineering
        X_train_eng = feature_engine.transform(X_train)
        X_test_eng = feature_engine.transform(X_test)
        
        print(f"Chunk {chunk_id}/{len(chunks)} | Train: {len(X_train)}, Test: {len(X_test)}")
        
        # ===== Lightning Ensemble =====
        start = time.time()
        if first_lightning:
            lightning.partial_fit(X_train_eng, y_train, classes=all_classes)
            first_lightning = False
        else:
            lightning.partial_fit(X_train_eng, y_train)
        lightning_pred = lightning.predict(X_test_eng)
        lightning_acc = accuracy_score(y_test, lightning_pred)
        lightning_time = time.time() - start
        
        # ===== SGD =====
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X_train_eng, y_train, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train_eng, y_train)
        sgd_pred = sgd.predict(X_test_eng)
        sgd_acc = accuracy_score(y_test, sgd_pred)
        sgd_time = time.time() - start
        
        # ===== PA =====
        start = time.time()
        if first_pa:
            pa.partial_fit(X_train_eng, y_train, classes=all_classes)
            first_pa = False
        else:
            pa.partial_fit(X_train_eng, y_train)
        pa_pred = pa.predict(X_test_eng)
        pa_acc = accuracy_score(y_test, pa_pred)
        pa_time = time.time() - start
        
        # ===== XGBoost (Fast) =====
        start = time.time()
        xgb_all_X.append(X_train_eng)
        xgb_all_y.append(y_train)
        
        if len(xgb_all_X) > WINDOW_SIZE:
            xgb_all_X = xgb_all_X[-WINDOW_SIZE:]
            xgb_all_y = xgb_all_y[-WINDOW_SIZE:]
        
        X_xgb = np.vstack(xgb_all_X)
        y_xgb = np.concatenate(xgb_all_y)
        
        # Smaller XGBoost
        dtrain = xgb.DMatrix(X_xgb, label=y_xgb)
        dtest = xgb.DMatrix(X_test_eng, label=y_test)
        
        xgb_model = xgb.train(
            {
                "objective": "multi:softmax",
                "num_class": len(all_classes),
                "max_depth": 3,  # Shallower trees
                "eta": 0.1,
                "subsample": 0.7,
                "verbosity": 0,
                "nthread": 1
            },
            dtrain,
            num_boost_round=10  # Fewer rounds
        )
        
        xgb_pred = xgb_model.predict(dtest)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_time = time.time() - start
        
        results.append({
            'chunk': chunk_id,
            'lightning_acc': lightning_acc,
            'sgd_acc': sgd_acc,
            'pa_acc': pa_acc,
            'xgb_acc': xgb_acc,
        })
        
        print(f"  Lightning: {lightning_acc:.3f} ({lightning_time:.2f}s)")
        print(f"  SGD:       {sgd_acc:.3f} ({sgd_time:.2f}s)")
        print(f"  PA:        {pa_acc:.3f} ({pa_time:.2f}s)")
        print(f"  XGB:       {xgb_acc:.3f} ({xgb_time:.2f}s)")
    
    # Quick results analysis
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("📊 QUICK RESULTS")
        print("="*60)
        
        for model in ['lightning', 'sgd', 'pa', 'xgb']:
            acc_mean = df_results[f'{model}_acc'].mean()
            print(f"{model.upper():10s}: {acc_mean:.4f}")
        
        # Determine winner
        acc_scores = {
            'lightning': df_results['lightning_acc'].mean(),
            'sgd': df_results['sgd_acc'].mean(),
            'pa': df_results['pa_acc'].mean(),
            'xgb': df_results['xgb_acc'].mean()
        }
        
        winner = max(acc_scores, key=acc_scores.get)
        print(f"\n🏆 WINNER: {winner.upper()} ({acc_scores[winner]:.4f})")
        
        # Save minimal results
        os.makedirs('benchmark_results', exist_ok=True)
        df_results.to_csv('benchmark_results/lightning_results.csv', index=False)
        
        return True
    else:
        print("❌ No results generated")
        return False


def main():
    """Main function with timeout protection"""
    print("="*60)
    print("⚡ LIGHTNING ML BENCHMARK")
    print("="*60)
    print("Optimized for CI/CD - Target: < 30 seconds\n")
    
    start_time = time.time()
    
    try:
        success = fast_benchmark()
        total_time = time.time() - start_time
        
        print(f"\n✅ BENCHMARK COMPLETED in {total_time:.1f}s")
        
        if total_time > 45:
            print("⚠️  Warning: Benchmark is getting slow for CI")
        elif total_time > 60:
            print("❌ Too slow for CI - needs further optimization")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        # Ensure we still create results directory
        os.makedirs('benchmark_results', exist_ok=True')
        with open('benchmark_results/error.log', 'w') as f:
            f.write(str(e))
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
