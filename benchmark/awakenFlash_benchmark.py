#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGHTNING NON-LOGIC ENSEMBLE - FAST & ACCURATE
Optimized for speed while maintaining accuracy
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ================= Helper functions ===================
def load_data(n_chunks=10, chunk_size=8000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, min(len(X_all), n_chunks * chunk_size), chunk_size)]
    return chunks[:n_chunks], np.unique(y_all)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# ================= Lightning Fast Ensemble ===================
class LightningEnsemble:
    def __init__(self):
        # Ultra-fast models with optimized parameters
        self.rf = RandomForestClassifier(
            n_estimators=80,           # Reduced from 120
            max_depth=15,              # Reduced from 25
            min_samples_split=10,      # More aggressive
            min_samples_leaf=4,        # More aggressive  
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,                 # Use all cores
            verbose=0
        )
        self.hgb = HistGradientBoostingClassifier(  # Much faster than GradientBoosting
            max_iter=100,              # Reduced iterations
            max_depth=10,              # Reasonable depth
            learning_rate=0.1,
            min_samples_leaf=20,       # More aggressive
            max_bins=128,              # Reduced for speed
            random_state=42,
            verbose=0
        )
        self.lr = LogisticRegression(
            max_iter=500,              # Reduced iterations
            C=1.0,
            solver='lbfgs',
            multi_class='auto',
            random_state=42,
            n_jobs=-1                  # Use all cores
        )
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        # Train all models in parallel using joblib
        from joblib import Parallel, delayed
        
        def train_rf(X, y):
            return self.rf.fit(X, y)
        
        def train_hgb(X, y):
            return self.hgb.fit(X, y)
        
        def train_lr(X, y):
            return self.lr.fit(X, y)
        
        # Parallel training
        start_time = time.time()
        results = Parallel(n_jobs=3)(
            delayed(train_rf)(X, y),
            delayed(train_hgb)(X, y),
            delayed(train_lr)(X, y)
        )
        
        self.rf, self.hgb, self.lr = results
        training_time = time.time() - start_time
        return training_time

    def predict(self, X):
        # Parallel prediction
        from joblib import Parallel, delayed
        
        def predict_rf(X):
            return self.rf.predict_proba(X)
        
        def predict_hgb(X):
            return self.hgb.predict_proba(X)
        
        def predict_lr(X):
            return self.lr.predict_proba(X)
        
        # Get probabilities in parallel
        rf_proba, hgb_proba, lr_proba = Parallel(n_jobs=3)(
            delayed(predict_rf)(X),
            delayed(predict_hgb)(X),
            delayed(predict_lr)(X)
        )
        
        # Optimized weighted voting
        weighted_proba = (2.0 * rf_proba + 3.0 * hgb_proba + 1.0 * lr_proba) / 6.0
        return self.classes_[np.argmax(weighted_proba, axis=1)]

# ================= Fast XGBoost ===================
class FastXGBoost:
    def __init__(self):
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        dtrain = xgb.DMatrix(X, label=y)
        
        # Fast XGBoost parameters
        self.model = xgb.train({
            "objective": "multi:softmax", 
            "num_class": len(self.classes_),
            "max_depth": 6,              # Reduced depth
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "tree_method": "hist",       # Faster tree method
            "verbosity": 0
        }, dtrain, num_boost_round=20)   # Reduced rounds
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest).astype(int)

# ================= Speed-Optimized Benchmark ===================
def speed_optimized_benchmark(chunks, all_classes):
    ensemble_wins = 0
    ensemble_scores = []
    xgb_scores = []
    ensemble_times = []
    xgb_times = []

    print("⚡ SPEED-OPTIMIZED BENCHMARK: LIGHTNING ENSEMBLE vs XGBOOST")
    print("=" * 65)
    print(f"Dataset: {len(chunks)} chunks, {len(all_classes)} classes")
    print("=" * 65)

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        print(f"\n🎯 Chunk {chunk_id}/{len(chunks)}")
        print("-" * 40)
        
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # Lightning Ensemble
        try:
            start_time = time.time()
            ensemble = LightningEnsemble()
            training_time = ensemble.fit(X_train, y_train)
            ensemble_pred = ensemble.predict(X_test)
            ensemble_acc = compute_accuracy(y_test, ensemble_pred)
            total_time = time.time() - start_time
            ensemble_scores.append(ensemble_acc)
            ensemble_times.append(total_time)
            
            print(f"  ⚡ Ensemble: Acc={ensemble_acc:.3f}, Time={total_time:.2f}s")
            print(f"    (Training: {training_time:.2f}s)")
        except Exception as e:
            print(f"  ❌ Ensemble failed: {e}")
            ensemble_acc = 0.0
            ensemble_scores.append(0.0)
            ensemble_times.append(0.0)

        # Fast XGBoost
        try:
            start_time = time.time()
            xgb_model = FastXGBoost()
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_acc = compute_accuracy(y_test, xgb_pred)
            xgb_time = time.time() - start_time
            xgb_scores.append(xgb_acc)
            xgb_times.append(xgb_time)
            
            print(f"  🚀 XGBoost:   Acc={xgb_acc:.3f}, Time={xgb_time:.2f}s")
        except Exception as e:
            print(f"  ❌ XGBoost failed: {e}")
            xgb_acc = 0.0
            xgb_scores.append(0.0)
            xgb_times.append(0.0)

        # Performance comparison
        accuracy_diff = ensemble_acc - xgb_acc
        time_ratio = ensemble_times[-1] / xgb_times[-1] if xgb_times[-1] > 0 else float('inf')
        
        if accuracy_diff > 0.01:
            ensemble_wins += 1
            result = "✅ ENSEMBLE DOMINATES"
            symbol = "🏆"
            color = "🟢"
        elif accuracy_diff > 0:
            ensemble_wins += 1
            result = "⚡ ENSEMBLE LEADS" 
            symbol = "🎯"
            color = "🟡"
        elif abs(accuracy_diff) <= 0.01:
            result = "⚖️  TOO CLOSE"
            symbol = "➖"
            color = "🔵"
        else:
            result = "🔥 XGBOOST LEADS"
            symbol = "⚠️"
            color = "🔴"

        print(f"  {color} {result} | Acc Diff: {accuracy_diff:+.3f}")
        print(f"  ⏱️  Speed Ratio: {time_ratio:.1f}x {'slower' if time_ratio > 1 else 'faster'}")

    # Final Results
    print("\n" + "=" * 65)
    print("🎉 FINAL RESULTS - SPEED OPTIMIZED")
    print("=" * 65)
    
    valid_ensemble = [s for s in ensemble_scores if s > 0]
    valid_xgb = [s for s in xgb_scores if s > 0]
    
    if valid_ensemble and valid_xgb:
        avg_ensemble = np.mean(valid_ensemble)
        avg_xgb = np.mean(valid_xgb)
        avg_ensemble_time = np.mean([t for t, s in zip(ensemble_times, ensemble_scores) if s > 0])
        avg_xgb_time = np.mean([t for t, s in zip(xgb_times, xgb_scores) if s > 0])
        
        print(f"📊 ACCURACY COMPARISON:")
        print(f"   ⚡ Lightning Ensemble: {avg_ensemble:.3f}")
        print(f"   🚀 XGBoost:           {avg_xgb:.3f}")
        print(f"   📈 Accuracy Advantage: {avg_ensemble - avg_xgb:+.3f}")
        
        print(f"⏱️  SPEED COMPARISON:")
        print(f"   ⚡ Ensemble: {avg_ensemble_time:.2f}s per chunk")
        print(f"   🚀 XGBoost:  {avg_xgb_time:.2f}s per chunk")
        speed_improvement = avg_xgb_time / avg_ensemble_time
        print(f"   💨 Speed Ratio: {speed_improvement:.1f}x {'faster' if speed_improvement > 1 else 'slower'}")
        
        print(f"🎯 PERFORMANCE SUMMARY:")
        print(f"   ✅ Ensemble wins: {ensemble_wins}/{len(chunks)} chunks")
        print(f"   📊 Win rate: {ensemble_wins/len(chunks)*100:.1f}%")
        
        # Performance assessment
        print("\n" + "⭐" * 65)
        if avg_ensemble > avg_xgb and avg_ensemble_time < avg_xgb_time * 5:  # Within 5x speed
            print("🎉 🏆 LIGHTNING ENSEMBLE WINS - FAST & ACCURATE! 🏆")
            print("   ✨ Best of both worlds - accuracy and speed!")
        elif avg_ensemble > avg_xgb:
            print("🎯 🥇 ENSEMBLE WINS - ACCURATE BUT SLOWER 🥇")
            print("   💪 Superior accuracy worth the wait!")
        elif avg_ensemble_time < avg_xgb_time:
            print("🚀 🥈 XGBoost wins accuracy but Ensemble wins speed! 🥈")
            print("   ⚡ Lightning fast with competitive accuracy!")
        else:
            print("🔥 XGBoost maintains both speed and accuracy advantage")
        print("⭐" * 65)
        
    else:
        print("❌ Benchmark incomplete - not enough successful runs")

# ================= Ultra-Fast Single Model ===================
class UltraFastModel:
    """Single model optimized for maximum speed"""
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=8,
            learning_rate=0.15,
            min_samples_leaf=30,
            max_bins=64,
            random_state=42,
            verbose=0
        )
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

# ================= Quick Speed Test ===================
def quick_speed_test(chunks, all_classes):
    """Ultra-fast benchmark for speed testing"""
    print("\n🚀 ULTRA-FAST SPEED TEST")
    print("=" * 50)
    
    ensemble_times = []
    xgb_times = []
    single_times = []
    
    ensemble_accs = []
    xgb_accs = [] 
    single_accs = []
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks[:4], 1):  # Test only 4 chunks
        print(f"Chunk {chunk_id}...", end=" ")
        
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # Ultra-fast single model
        start_time = time.time()
        single = UltraFastModel()
        single.fit(X_train, y_train)
        single_pred = single.predict(X_test)
        single_acc = compute_accuracy(y_test, single_pred)
        single_time = time.time() - start_time
        single_times.append(single_time)
        single_accs.append(single_acc)

        # Lightning Ensemble
        start_time = time.time()
        ensemble = LightningEnsemble()
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_acc = compute_accuracy(y_test, ensemble_pred)
        ensemble_time = time.time() - start_time
        ensemble_times.append(ensemble_time)
        ensemble_accs.append(ensemble_acc)

        # Fast XGBoost
        start_time = time.time()
        xgb_model = FastXGBoost()
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = compute_accuracy(y_test, xgb_pred)
        xgb_time = time.time() - start_time
        xgb_times.append(xgb_time)
        xgb_accs.append(xgb_acc)

        print(f"S:{single_time:.2f}s/E:{ensemble_time:.2f}s/X:{xgb_time:.2f}s")

    # Results
    print(f"\n📊 SPEED TEST RESULTS (4 chunks):")
    print(f"   ⚡ Single Model:    {np.mean(single_times):.2f}s, Acc: {np.mean(single_accs):.3f}")
    print(f"   ⚡ Ensemble:        {np.mean(ensemble_times):.2f}s, Acc: {np.mean(ensemble_accs):.3f}")
    print(f"   🚀 XGBoost:         {np.mean(xgb_times):.2f}s, Acc: {np.mean(xgb_accs):.3f}")

# ================= Main ===================
if __name__ == "__main__":
    chunks, all_classes = load_data(n_chunks=8, chunk_size=8000)
    
    print("⚡ LIGHTNING NON-LOGIC ENSEMBLE ACTIVATED")
    print("💨 Optimized for Maximum Speed + Accuracy")
    print("🏆 Ready to Compete with XGBoost on All Fronts")
    print()
    
    # Run speed-optimized benchmark
    speed_optimized_benchmark(chunks, all_classes)
    
    # Quick speed test
    quick_speed_test(chunks, all_classes)
    
    print("\n" + "=" * 65)
    print("💡 Use LightningEnsemble() for best speed/accuracy balance")
    print("💡 Use UltraFastModel() for maximum speed")
    print("=" * 65)
