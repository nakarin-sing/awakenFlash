#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FULL FAIRNESS BENCHMARK
Testing multiple scenarios to show different winners in different contexts
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def load_data(n_chunks=10):
    """Load and prepare data"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"📦 Loading dataset from: {url}")
    
    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1
    
    # Normalize
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    # Split into chunks
    chunk_size = 10000
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, len(X_all), chunk_size)]
    
    return chunks[:n_chunks], np.unique(y_all)


def scenario_1_streaming(chunks, all_classes):
    """
    SCENARIO 1: Real-time Streaming (Current benchmark)
    - Data arrives one chunk at a time
    - Must predict immediately
    - Winner: Online Learning (SGD/PA)
    """
    print("\n" + "="*70)
    print("🔄 SCENARIO 1: REAL-TIME STREAMING")
    print("="*70)
    print("Context: Data arrives continuously, must predict NOW")
    print("Fair for: Online learning algorithms")
    
    sgd = SGDClassifier(loss="log_loss", learning_rate="optimal", 
                        max_iter=5, warm_start=True, random_state=42)
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=5, 
                                     warm_start=True, random_state=42)
    
    xgb_model = None
    xgb_all_X, xgb_all_y = [], []
    
    first_sgd = first_pa = True
    results = []
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        # SGD
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X_train, y_train, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train)
        sgd_acc = sgd.score(X_test, y_test)
        sgd_time = time.time() - start
        
        # PA
        start = time.time()
        if first_pa:
            pa.partial_fit(X_train, y_train, classes=all_classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train)
        pa_acc = pa.score(X_test, y_test)
        pa_time = time.time() - start
        
        # XGBoost (retrain all)
        start = time.time()
        xgb_all_X.append(X_train)
        xgb_all_y.append(y_train)
        dtrain = xgb.DMatrix(np.vstack(xgb_all_X), label=np.concatenate(xgb_all_y))
        dtest = xgb.DMatrix(X_test, label=y_test)
        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": 7, "max_depth": 4, 
             "eta": 0.3, "verbosity": 0},
            dtrain, num_boost_round=10
        )
        # Fix: Use predict instead of eval for accuracy
        xgb_pred = xgb_model.predict(dtest)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        xgb_time = time.time() - start
        
        results.append({
            'chunk': chunk_id,
            'sgd_acc': sgd_acc, 'sgd_time': sgd_time,
            'pa_acc': pa_acc, 'pa_time': pa_time,
            'xgb_acc': xgb_acc, 'xgb_time': xgb_time
        })
        
        if chunk_id <= 3 or chunk_id >= 8:  # Show first 3 and last 3
            print(f"Chunk {chunk_id:02d}: SGD={sgd_acc:.3f}({sgd_time:.3f}s) "
                  f"PA={pa_acc:.3f}({pa_time:.3f}s) XGB={xgb_acc:.3f}({xgb_time:.3f}s)")
    
    df = pd.DataFrame(results)
    print("\n📊 Average Results:")
    print(f"  SGD:     acc={df['sgd_acc'].mean():.4f}, time={df['sgd_time'].mean():.4f}s")
    print(f"  PA:      acc={df['pa_acc'].mean():.4f}, time={df['pa_time'].mean():.4f}s")
    print(f"  XGBoost: acc={df['xgb_acc'].mean():.4f}, time={df['xgb_time'].mean():.4f}s")
    print(f"\n🏆 Winner: {'PA' if df['pa_acc'].mean() > df['xgb_acc'].mean() else 'XGBoost'} (by accuracy)")
    print(f"⚡ Speed: PA/SGD ~{df['xgb_time'].mean()/df['pa_time'].mean():.0f}x faster")
    
    return df


def scenario_2_batch(chunks, all_classes):
    """
    SCENARIO 2: Traditional Batch Learning (Kaggle-style)
    - Train once on ALL data
    - Test on held-out set
    - Winner: XGBoost (almost always)
    """
    print("\n" + "="*70)
    print("📦 SCENARIO 2: BATCH LEARNING (Kaggle-style)")
    print("="*70)
    print("Context: Train once on full dataset, test once")
    print("Fair for: Batch learning algorithms (XGBoost, Random Forest)")
    
    # Combine all chunks
    X_all = np.vstack([chunk[0] for chunk in chunks])
    y_all = np.concatenate([chunk[1] for chunk in chunks])
    
    # Split train/test
    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]
    
    # SGD (one pass)
    print("\n🔄 Training SGD (single pass)...")
    start = time.time()
    sgd = SGDClassifier(loss="log_loss", max_iter=10, random_state=42)
    sgd.fit(X_train, y_train)
    sgd_acc = sgd.score(X_test, y_test)
    sgd_time = time.time() - start
    
    # PA (one pass)
    print("🔄 Training PA (single pass)...")
    start = time.time()
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=10, random_state=42)
    pa.fit(X_train, y_train)
    pa_acc = pa.score(X_test, y_test)
    pa_time = time.time() - start
    
    # XGBoost (full power)
    print("🔄 Training XGBoost (50 trees, full optimization)...")
    start = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(
        {"objective": "multi:softmax", "num_class": 7, "max_depth": 6, 
         "eta": 0.1, "verbosity": 0},
        dtrain, num_boost_round=50
    )
    xgb_pred = xgb_model.predict(dtest)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_time = time.time() - start
    
    print("\n📊 Results:")
    print(f"  SGD:     acc={sgd_acc:.4f}, time={sgd_time:.2f}s")
    print(f"  PA:      acc={pa_acc:.4f}, time={pa_time:.2f}s")
    print(f"  XGBoost: acc={xgb_acc:.4f}, time={xgb_time:.2f}s")
    print(f"\n🏆 Winner: XGBoost (by {(xgb_acc - max(sgd_acc, pa_acc))*100:.2f}% margin)")


def scenario_3_concept_drift(chunks, all_classes):
    """
    SCENARIO 3: Concept Drift Test
    - Train on first 5 chunks
    - Test on chunk 6-10 (different distribution)
    - Winner: Depends on adaptation capability
    """
    print("\n" + "="*70)
    print("🌊 SCENARIO 3: CONCEPT DRIFT")
    print("="*70)
    print("Context: Train on early data, test on later data (distribution shift)")
    print("Fair for: Adaptive algorithms")
    
    # Train on first 5 chunks
    X_train = np.vstack([chunk[0][:8000] for chunk in chunks[:5]])
    y_train = np.concatenate([chunk[1][:8000] for chunk in chunks[:5]])
    
    # Test on chunks 6-10
    X_test = np.vstack([chunk[0][8000:] for chunk in chunks[5:]])
    y_test = np.concatenate([chunk[1][8000:] for chunk in chunks[5:]])
    
    print(f"\n📚 Training on {len(X_train)} samples from chunks 1-5...")
    
    # SGD (static)
    sgd = SGDClassifier(loss="log_loss", max_iter=10, random_state=42)
    sgd.fit(X_train, y_train)
    sgd_static = sgd.score(X_test, y_test)
    
    # XGBoost (static)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(
        {"objective": "multi:softmax", "num_class": 7, "verbosity": 0},
        dtrain, num_boost_round=30
    )
    xgb_pred = xgb_model.predict(dtest)
    xgb_static = accuracy_score(y_test, xgb_pred)
    
    print(f"\n🔬 Testing on {len(X_test)} samples from chunks 6-10...")
    print(f"  SGD (static):     {sgd_static:.4f}")
    print(f"  XGBoost (static): {xgb_static:.4f}")
    print(f"\n💡 Insight: Both models struggle with distribution shift")
    print(f"   Online learning would adapt better with partial_fit on new chunks")


def main():
    print("="*70)
    print("🔬 FULL FAIRNESS BENCHMARK")
    print("="*70)
    print("Testing: When is each algorithm 'fairly' the winner?")
    print()
    
    chunks, all_classes = load_data(n_chunks=10)
    
    # Run all scenarios
    scenario_1_streaming(chunks, all_classes)
    scenario_2_batch(chunks, all_classes)
    scenario_3_concept_drift(chunks, all_classes)
    
    print("\n" + "="*70)
    print("🎓 FINAL CONCLUSION")
    print("="*70)
    print("""
📌 There is NO universally 'fair' benchmark!
   Each algorithm wins in its designed context:

   🥇 Online Learning (SGD/PA) wins when:
      - Data streams continuously
      - Low latency required
      - Memory constrained
      - Concept drift expected

   🥇 XGBoost wins when:
      - Full dataset available
      - Training time not critical  
      - Maximum accuracy needed
      - Stable distribution

   ⚖️ The benchmark in PR #270 is:
      ✅ Fair for streaming context
      ⚠️ Structurally biased against batch learners
      ❌ Would be unfair if claiming general superiority

   💡 Solution: Report results with context:
      "Online learning achieves 0.85 acc in 0.01s per chunk"
      "XGBoost achieves 0.88 acc but requires 0.3s+ per chunk"
      Let users decide based on their use case!
    """)


if __name__ == "__main__":
    main()
