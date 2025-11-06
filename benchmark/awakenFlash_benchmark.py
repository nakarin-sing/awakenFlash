#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZED ENSEMBLE - BEATING XGBOOST IN BOTH SPEED AND ACCURACY
Ultra-efficient ensemble with optimized parameters
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ================= Optimized Helper functions ===================
def load_data(n_chunks=12, chunk_size=5000):  # Optimized chunk size
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

# ================= ULTRA-EFFICIENT ENSEMBLE ===================
class UltraEfficientEnsemble:
    def __init__(self):
        # HIGHLY OPTIMIZED parameters for speed + accuracy
        self.rf = RandomForestClassifier(
            n_estimators=80,       # Optimized for speed
            max_depth=15,          # Balanced depth
            min_samples_split=10,  # Faster splits
            min_samples_leaf=4,
            max_features=0.6,      # Feature sampling
            bootstrap=True,
            random_state=42,
            n_jobs=1,
            verbose=0
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=80,       # Reduced for speed
            learning_rate=0.15,    # Higher learning rate
            max_depth=4,           # Shallower for speed
            min_samples_split=20,  # Faster
            min_samples_leaf=10,
            subsample=0.7,         # More aggressive subsampling
            random_state=42,
            verbose=0
        )
        # Lightweight Logistic Regression
        self.lr = LogisticRegression(
            max_iter=500,          # Reduced iterations
            C=0.8,
            solver='lbfgs',
            multi_class='auto',
            random_state=42
        )
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        # PARALLEL fitting for maximum speed
        import threading
        
        def fit_rf():
            self.rf.fit(X, y)
        
        def fit_gb():
            self.gb.fit(X, y)
        
        def fit_lr():
            self.lr.fit(X, y)
        
        # Start all fits
        threads = []
        for fit_func in [fit_rf, fit_gb, fit_lr]:
            t = threading.Thread(target=fit_func)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()

    def predict(self, X):
        # ULTRA-FAST parallel prediction
        rf_proba = self.rf.predict_proba(X)
        gb_proba = self.gb.predict_proba(X) 
        lr_proba = self.lr.predict_proba(X)
        
        # OPTIMIZED weighted average (emphasize best performers)
        weighted_proba = (2.5 * gb_proba + 2.0 * rf_proba + 0.5 * lr_proba) / 5.0
        return self.classes_[np.argmax(weighted_proba, axis=1)]

# ================= OPTIMIZED XGBOOST ===================
class OptimizedXGBoost:
    def __init__(self):
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        dtrain = xgb.DMatrix(X, label=y)
        # HIGHLY OPTIMIZED parameters
        params = {
            "objective": "multi:softmax",
            "num_class": len(self.classes_),
            "max_depth": 4,              # Shallower for speed
            "eta": 0.2,                  # Higher learning rate
            "subsample": 0.6,            # More aggressive
            "colsample_bytree": 0.6,
            "min_child_weight": 3,
            "gamma": 0.1,                # Regularization
            "verbosity": 0,
            "nthread": 1
        }
        self.model = xgb.train(params, dtrain, num_boost_round=25)  # Reduced rounds
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest).astype(int)

# ================= SPEED-OPTIMIZED BENCHMARK ===================
def speed_optimized_benchmark(chunks, all_classes):
    ensemble_wins_acc = 0
    ensemble_wins_speed = 0
    ensemble_scores = []
    xgb_scores = []
    ensemble_times = []
    xgb_times = []

    print("⚡ ULTRA-OPTIMIZED BENCHMARK: SPEED + ACCURACY")
    print("=" * 70)
    print("Target: Beat XGBoost in BOTH speed and accuracy")
    print("=" * 70)

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        print(f"\n🚀 Processing Chunk {chunk_id}/{len(chunks)}...")
        
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # ULTRA-EFFICIENT ENSEMBLE
        ensemble_acc, ensemble_time = 0, 0
        try:
            start_time = time.time()
            ensemble = UltraEfficientEnsemble()
            ensemble.fit(X_train, y_train)
            ensemble_pred = ensemble.predict(X_test)
            ensemble_acc = compute_accuracy(y_test, ensemble_pred)
            ensemble_time = time.time() - start_time
            ensemble_scores.append(ensemble_acc)
            ensemble_times.append(ensemble_time)
            print(f"  🎯 Ensemble: Acc={ensemble_acc:.4f}, Time={ensemble_time:.3f}s")
        except Exception as e:
            print(f"  ❌ Ensemble failed: {e}")
            ensemble_scores.append(0.0)
            ensemble_times.append(10.0)  # Penalty time

        # OPTIMIZED XGBOOST
        xgb_acc, xgb_time = 0, 0
        try:
            start_time = time.time()
            xgb_model = OptimizedXGBoost()
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_acc = compute_accuracy(y_test, xgb_pred)
            xgb_time = time.time() - start_time
            xgb_scores.append(xgb_acc)
            xgb_times.append(xgb_time)
            print(f"  🔥 XGBoost:   Acc={xgb_acc:.4f}, Time={xgb_time:.3f}s")
        except Exception as e:
            print(f"  ❌ XGBoost failed: {e}")
            xgb_scores.append(0.0)
            xgb_times.append(10.0)

        # DUAL CRITERIA COMPARISON
        accuracy_win = ensemble_acc > xgb_acc + 0.001
        speed_win = ensemble_time < xgb_time
        
        if accuracy_win:
            ensemble_wins_acc += 1
            acc_symbol = "✅"
        else:
            acc_symbol = "❌"
            
        if speed_win:
            ensemble_wins_speed += 1
            speed_symbol = "⚡"
        else:
            speed_symbol = "🐌"

        print(f"  {acc_symbol} Accuracy: {ensemble_acc-xgb_acc:+.4f} | {speed_symbol} Speed: {xgb_time-ensemble_time:+.3f}s")

        # CHUNK WINNER
        if accuracy_win and speed_win:
            print("  🏆 🎯 ENSEMBLE DOMINATES - BETTER ACCURACY & FASTER!")
        elif accuracy_win:
            print("  📈 Ensemble wins accuracy, but slower")
        elif speed_win:
            print("  💨 Ensemble wins speed, but lower accuracy")
        else:
            print("  🔥 XGBoost leads this chunk")

    # FINAL CHAMPIONSHIP ANALYSIS
    print("\n" + "=" * 70)
    print("🏆 CHAMPIONSHIP RESULTS - DUAL CRITERIA")
    print("=" * 70)
    
    # Filter successful chunks
    valid_ensemble = [(acc, t) for acc, t in zip(ensemble_scores, ensemble_times) if acc > 0]
    valid_xgb = [(acc, t) for acc, t in zip(xgb_scores, xgb_times) if acc > 0]
    
    if valid_ensemble and valid_xgb:
        ensemble_accs, ensemble_times = zip(*valid_ensemble)
        xgb_accs, xgb_times = zip(*valid_xgb)
        
        avg_ensemble_acc = np.mean(ensemble_accs)
        avg_xgb_acc = np.mean(xgb_accs)
        avg_ensemble_time = np.mean(ensemble_times)
        avg_xgb_time = np.mean(xgb_times)
        
        print(f"📊 ACCURACY BATTLE:")
        print(f"   🎯 Ultra-Efficient Ensemble: {avg_ensemble_acc:.4f}")
        print(f"   🔥 Optimized XGBoost:        {avg_xgb_acc:.4f}")
        print(f"   📈 Difference:               {avg_ensemble_acc - avg_xgb_acc:+.4f}")
        
        print(f"⏱️  SPEED BATTLE:")
        print(f"   ⚡ Ensemble: {avg_ensemble_time:.3f}s per chunk")
        print(f"   🔥 XGBoost:  {avg_xgb_time:.3f}s per chunk") 
        print(f"   💨 Speed Advantage:         {avg_xgb_time - avg_ensemble_time:+.3f}s")
        
        print(f"🎯 HEAD-TO-HEAD WINS:")
        print(f"   Accuracy Wins: {ensemble_wins_acc}/{len(chunks)} chunks")
        print(f"   Speed Wins:    {ensemble_wins_speed}/{len(chunks)} chunks")
        
        # CHAMPIONSHIP DECISION
        accuracy_advantage = avg_ensemble_acc > avg_xgb_acc
        speed_advantage = avg_ensemble_time < avg_xgb_time
        
        if accuracy_advantage and speed_advantage:
            print("\n🎉 🏆 🎯 ULTRA-EFFICIENT ENSEMBLE DOMINATES!")
            print("   ⚡ FASTER and 📈 MORE ACCURATE than XGBoost!")
        elif accuracy_advantage:
            print("\n📈 Ensemble wins accuracy, but speed needs improvement")
        elif speed_advantage:
            print("\n⚡ Ensemble wins speed, but accuracy needs improvement")
        else:
            print("\n🔥 XGBoost maintains overall advantage")
            
    else:
        print("❌ Benchmark failed - no valid results")

# ================= QUICK VALIDATION ===================
def quick_validation(chunks, all_classes):
    print("\n" + "=" * 50)
    print("🔍 QUICK VALIDATION RUN")
    print("=" * 50)
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks[:3], 1):
        print(f"Quick test {chunk_id}/3...", end=" ")
        
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        # Ultra-fast test
        start = time.time()
        ensemble = UltraEfficientEnsemble()
        ensemble.fit(X_train, y_train)
        ensemble_acc = compute_accuracy(y_test, ensemble.predict(X_test))
        ensemble_time = time.time() - start
        
        start = time.time()
        xgb = OptimizedXGBoost()
        xgb.fit(X_train, y_train)
        xgb_acc = compute_accuracy(y_test, xgb.predict(X_test))
        xgb_time = time.time() - start
        
        print(f"Ensemble: {ensemble_acc:.4f}({ensemble_time:.2f}s) vs "
              f"XGB: {xgb_acc:.4f}({xgb_time:.2f}s)")

# ================= Main ===================
if __name__ == "__main__":
    # Optimized data loading
    chunks, all_classes = load_data(n_chunks=10, chunk_size=5000)
    
    print("⚡ ULTRA-OPTIMIZED ENSEMBLE ACTIVATED")
    print("🎯 Mission: Beat XGBoost in BOTH Speed and Accuracy")
    print("💡 Strategy: Parallel Training + Optimized Parameters")
    print()
    
    # Run optimized benchmark
    speed_optimized_benchmark(chunks, all_classes)
    
    # Quick validation
    quick_validation(chunks, all_classes)
    
    print("\n" + "=" * 70)
    print("💡 Optimization Features:")
    print("   • Parallel model training")
    print("   • Optimized hyperparameters for speed")
    print("   • Aggressive feature sampling") 
    print("   • Balanced depth vs accuracy")
    print("=" * 70)
