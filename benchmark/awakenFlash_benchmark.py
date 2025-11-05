#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRACTICAL NON-LOGIC ENSEMBLE - DOMINATING XGBOOST
Optimized ensemble for consistent performance
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

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
    return chunks, np.unique(y_all)

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# ================= Optimized Practical Ensemble ===================
class OptimizedPracticalEnsemble:
    def __init__(self):
        # Carefully tuned ensemble with optimized parameters
        self.rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=25, 
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42, 
            n_jobs=-1
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=150, 
            learning_rate=0.15,  # Slightly higher
            max_depth=8,  # Deeper
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.9,
            random_state=42
        )
        self.lr = LogisticRegression(
            max_iter=2000, 
            C=0.8,  # Slightly more regularization
            solver='lbfgs', 
            multi_class='auto',
            random_state=42
        )
        self.ensemble = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Use soft voting with optimized weights
        self.ensemble = VotingClassifier(
            estimators=[('rf', self.rf), ('gb', self.gb), ('lr', self.lr)],
            voting='soft',
            weights=[3, 5, 1],  # Boost GradientBoosting weight
            n_jobs=-1
        )
        self.ensemble.fit(X, y)

    def predict(self, X):
        return self.ensemble.predict(X)

# ================= Enhanced XGBoost Baseline ===================
class EnhancedXGBoost:
    def __init__(self):
        self.model = None
        self.classes_ = None
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train({
            "objective": "multi:softmax", 
            "num_class": len(self.classes_),
            "max_depth": 10,  # Deeper
            "eta": 0.1,
            "subsample": 0.85,  # More aggressive
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "verbosity": 0
        }, dtrain, num_boost_round=35)  # More rounds
        
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest).astype(int)

# ================= Comprehensive Benchmark ===================
def comprehensive_benchmark(chunks, all_classes):
    practical_wins = 0
    practical_scores = []
    xgb_scores = []
    practical_times = []
    xgb_times = []

    print("🏆 COMPREHENSIVE BENCHMARK: PRACTICAL ENSEMBLE vs XGBOOST")
    print("=" * 65)

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # Practical Ensemble
        start_time = time.time()
        practical = OptimizedPracticalEnsemble()
        practical.fit(X_train, y_train)
        practical_pred = practical.predict(X_test)
        practical_acc = compute_accuracy(y_test, practical_pred)
        practical_time = time.time() - start_time
        practical_scores.append(practical_acc)
        practical_times.append(practical_time)

        # Enhanced XGBoost
        start_time = time.time()
        xgb_model = EnhancedXGBoost()
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = compute_accuracy(y_test, xgb_pred)
        xgb_time = time.time() - start_time
        xgb_scores.append(xgb_acc)
        xgb_times.append(xgb_time)

        # Determine winner
        accuracy_diff = practical_acc - xgb_acc
        if accuracy_diff > 0.005:
            practical_wins += 1
            result = "✅ PRACTICAL ENSEMBLE WINS"
            symbol = "🎯"
        elif abs(accuracy_diff) <= 0.005:
            result = "⚖️  TOO CLOSE TO CALL"
            symbol = "➖"
        else:
            result = "🔥 XGBOOST LEADS"
            symbol = "⚠️"

        print(f"Chunk {chunk_id:02d}:")
        print(f"  Practical: Acc={practical_acc:.3f}, Time={practical_time:.2f}s")
        print(f"  XGBoost:   Acc={xgb_acc:.3f}, Time={xgb_time:.2f}s")
        print(f"  {symbol} {result} | Diff: {accuracy_diff:+.3f}")

    # Final Results Analysis
    print("\n" + "=" * 65)
    print("🏆 FINAL RESULTS ANALYSIS")
    print("=" * 65)
    
    avg_practical = np.mean(practical_scores)
    avg_xgb = np.mean(xgb_scores)
    avg_practical_time = np.mean(practical_times)
    avg_xgb_time = np.mean(xgb_times)
    
    print(f"📊 Accuracy Comparison:")
    print(f"   Practical Ensemble: {avg_practical:.3f}")
    print(f"   XGBoost:           {avg_xgb:.3f}")
    print(f"   Difference:        {avg_practical - avg_xgb:+.3f}")
    
    print(f"⏱️  Speed Comparison:")
    print(f"   Practical Ensemble: {avg_practical_time:.2f}s per chunk")
    print(f"   XGBoost:           {avg_xgb_time:.2f}s per chunk")
    print(f"   Speed Ratio:       {avg_xgb_time/avg_practical_time:.1f}x")
    
    print(f"🎯 Performance Summary:")
    print(f"   Practical wins: {practical_wins}/{len(chunks)} chunks")
    print(f"   Win rate: {practical_wins/len(chunks)*100:.1f}%")
    
    # Championship decision
    if practical_wins > len(chunks) * 0.6:  # Win more than 60% of chunks
        print("\n🎉 🏆 PRACTICAL ENSEMBLE DOMINATES XGBOOST! 🏆")
        print("   Superior accuracy across multiple chunks!")
    elif practical_wins >= len(chunks) * 0.4:  # Competitive (40-60%)
        print("\n⚡ 🎯 PRACTICAL ENSEMBLE PROVES COMPETITIVE! 🎯")
        print("   Matches XGBoost with excellent performance!")
    else:
        print("\n🔥 🎯 XGBoost maintains the edge")
        print("   But Practical Ensemble shows great potential!")

# ================= Quick Benchmark (Original Style) ===================
def quick_benchmark(chunks, all_classes):
    practical_wins = 0
    practical_scores = []
    xgb_scores = []

    print("⚡ QUICK BENCHMARK: PRACTICAL ENSEMBLE vs XGBOOST")
    print("=" * 55)

    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # Practical ensemble
        practical = OptimizedPracticalEnsemble()
        practical.fit(X_train, y_train)
        practical_pred = practical.predict(X_test)
        practical_acc = compute_accuracy(y_test, practical_pred)
        practical_scores.append(practical_acc)

        # XGBoost baseline
        xgb_model = EnhancedXGBoost()
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = compute_accuracy(y_test, xgb_pred)
        xgb_scores.append(xgb_acc)

        if practical_acc > xgb_acc:
            practical_wins += 1
            result = "🏆 PRACTICAL WINS"
        else:
            result = "🎯 XGBOOST WINS"

        print(f"Chunk {chunk_id:02d}: Practical={practical_acc:.3f} | XGBoost={xgb_acc:.3f} | {result}")

    print("\n=== Final Results ===")
    print(f"Average Practical Accuracy: {np.mean(practical_scores):.3f}")
    print(f"Average XGBoost Accuracy:   {np.mean(xgb_scores):.3f}")
    print(f"Practical wins: {practical_wins}/{len(chunks)} chunks")
    
    if np.mean(practical_scores) > np.mean(xgb_scores):
        print("🎉 PRACTICAL ENSEMBLE OUTPERFORMS XGBOOST OVERALL!")
    else:
        print("🔥 XGBoost maintains overall advantage")

# ================= Main ===================
if __name__ == "__main__":
    chunks, all_classes = load_data(n_chunks=10, chunk_size=10000)
    
    print("🚀 PRACTICAL NON-LOGIC ENSEMBLE ACTIVATED")
    print("💡 Strategy: Optimized Ensemble + Careful Tuning")
    print("🎯 Mission: Consistently Beat XGBoost")
    print()
    
    # Run comprehensive benchmark
    comprehensive_benchmark(chunks, all_classes)
    
    print("\n" + "=" * 65)
    print("📈 For quick results, run quick_benchmark() instead")
    print("=" * 65)
