#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED MULTI-SCENARIO ML BENCHMARK
Comprehensive fairness testing with multiple metrics and visualizations
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from scipy import stats
import tracemalloc
import warnings
warnings.filterwarnings('ignore')

# =========================
# ✅ Ensure optional dependencies
# =========================
try:
    import tabulate
except ImportError:
    os.system("pip install tabulate")
    import tabulate


class MemoryTracker:
    """Track memory usage"""
    def __init__(self):
        self.snapshots = []
        tracemalloc.start()
    
    def snapshot(self, label):
        current, peak = tracemalloc.get_traced_memory()
        self.snapshots.append({
            'label': label,
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024
        })
    
    def get_usage(self):
        return pd.DataFrame(self.snapshots)


def load_data(n_chunks=10, chunk_size=10000):
    """Load and prepare data"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print(f"📦 Loading dataset from UCI ML Repository...")
    
    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1
    
    print(f"   Dataset shape: {X_all.shape}")
    print(f"   Classes: {np.unique(y_all)}")
    print(f"   Normalizing features...")
    
    # Normalize
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    # Split into chunks
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, min(len(X_all), n_chunks * chunk_size), chunk_size)]
    
    print(f"   Created {len(chunks)} chunks of size {chunk_size}\n")
    return chunks[:n_chunks], np.unique(y_all)


def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def scenario_1_streaming(chunks, all_classes, memory_tracker):
    """
    SCENARIO 1: Real-time Streaming with Comprehensive Metrics
    """
    WINDOW_SIZE = 5  # XGBoost sliding window size
    
    print("\n" + "="*80)
    print("🔄 SCENARIO 1: REAL-TIME STREAMING (Online vs Batch)")
    print("="*80)
    print("Context: Data arrives chunk by chunk, predict immediately")
    print("Metrics: Accuracy, Precision, Recall, F1, Speed, Memory")
    print(f"XGBoost uses sliding window of {WINDOW_SIZE} chunks for fairness\n")
    
    # Initialize models
    sgd = SGDClassifier(
        loss="log_loss", 
        learning_rate="optimal",
        max_iter=10,
        warm_start=True, 
        random_state=42,
        n_jobs=-1
    )
    
    pa = PassiveAggressiveClassifier(
        C=0.01, 
        max_iter=10,
        warm_start=True,
        random_state=42,
        n_jobs=-1
    )
    
    # XGBoost with sliding window approach (fair compromise)
    xgb_params = {
        "objective": "multi:softmax",
        "num_class": 7,
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "nthread": -1
    }
    
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 5  # Keep last 5 chunks only (fair memory usage)
    
    first_sgd = first_pa = True
    results = []
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        # Split chunk
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        print(f"Chunk {chunk_id:02d} | Train: {len(X_train)}, Test: {len(X_test)}")
        
        # ===== SGD =====
        memory_tracker.snapshot(f'sgd_chunk{chunk_id}_start')
        start = time.time()
        
        if first_sgd:
            sgd.partial_fit(X_train, y_train, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train)
        
        sgd_pred = sgd.predict(X_test)
        sgd_metrics = compute_metrics(y_test, sgd_pred)
        sgd_time = time.time() - start
        memory_tracker.snapshot(f'sgd_chunk{chunk_id}_end')
        
        # ===== Passive-Aggressive =====
        memory_tracker.snapshot(f'pa_chunk{chunk_id}_start')
        start = time.time()
        
        if first_pa:
            pa.partial_fit(X_train, y_train, classes=all_classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train)
        
        pa_pred = pa.predict(X_test)
        pa_metrics = compute_metrics(y_test, pa_pred)
        pa_time = time.time() - start
        memory_tracker.snapshot(f'pa_chunk{chunk_id}_end')
        
        # ===== XGBoost (Sliding Window) =====
        memory_tracker.snapshot(f'xgb_chunk{chunk_id}_start')
        start = time.time()
        
        # Add current chunk to buffer
        xgb_all_X.append(X_train)
        xgb_all_y.append(y_train)
        
        # Keep only recent chunks (sliding window)
        if len(xgb_all_X) > WINDOW_SIZE:
            xgb_all_X = xgb_all_X[-WINDOW_SIZE:]
            xgb_all_y = xgb_all_y[-WINDOW_SIZE:]
        
        # Train on windowed data
        X_xgb_train = np.vstack(xgb_all_X)
        y_xgb_train = np.concatenate(xgb_all_y)
        
        dtrain = xgb.DMatrix(X_xgb_train, label=y_xgb_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train fresh model on window (not incremental, but fair)
        xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=20)
        
        xgb_pred = xgb_model.predict(dtest)
        xgb_metrics = compute_metrics(y_test, xgb_pred)
        xgb_time = time.time() - start
        memory_tracker.snapshot(f'xgb_chunk{chunk_id}_end')
        
        # Store results
        results.append({
            'chunk': chunk_id,
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
        
        # Print progress
        print(f"  SGD: acc={sgd_metrics['accuracy']:.3f} f1={sgd_metrics['f1']:.3f} t={sgd_time:.3f}s")
        print(f"  PA:  acc={pa_metrics['accuracy']:.3f} f1={pa_metrics['f1']:.3f} t={pa_time:.3f}s")
        print(f"  XGB: acc={xgb_metrics['accuracy']:.3f} f1={xgb_metrics['f1']:.3f} t={xgb_time:.3f}s")
        print()
    
    df_results = pd.DataFrame(results)
    
    # Statistical analysis
    print("\n📊 STATISTICAL SUMMARY")
    print("="*80)
    
    for model in ['sgd', 'pa', 'xgb']:
        acc_mean = df_results[f'{model}_acc'].mean()
        acc_std = df_results[f'{model}_acc'].std()
        f1_mean = df_results[f'{model}_f1'].mean()
        time_mean = df_results[f'{model}_time'].mean()
        
        print(f"\n{model.upper():5s}: Accuracy={acc_mean:.4f}±{acc_std:.4f} | "
              f"F1={f1_mean:.4f} | Time={time_mean:.4f}s")
    
    # Statistical significance test
    print("\n📈 STATISTICAL SIGNIFICANCE (Paired t-test on accuracy)")
    print("-"*80)
    
    # PA vs SGD
    t_stat, p_val = stats.ttest_rel(df_results['pa_acc'], df_results['sgd_acc'])
    sig = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
    print(f"PA vs SGD:     t={t_stat:.3f}, p={p_val:.4f} {sig}")
    
    # XGB vs PA
    t_stat, p_val = stats.ttest_rel(df_results['xgb_acc'], df_results['pa_acc'])
    sig = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
    print(f"XGB vs PA:     t={t_stat:.3f}, p={p_val:.4f} {sig}")
    
    # XGB vs SGD
    t_stat, p_val = stats.ttest_rel(df_results['xgb_acc'], df_results['sgd_acc'])
    sig = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
    print(f"XGB vs SGD:    t={t_stat:.3f}, p={p_val:.4f} {sig}")
    
    return df_results


def scenario_2_batch(chunks, all_classes, memory_tracker):
    """
    SCENARIO 2: Traditional Batch Learning
    """
    print("\n" + "="*80)
    print("📦 SCENARIO 2: BATCH LEARNING (Traditional ML Pipeline)")
    print("="*80)
    print("Context: Train once on full dataset, test on held-out set")
    print("This is the 'Kaggle-style' approach\n")
    
    # Combine all chunks
    X_all = np.vstack([chunk[0] for chunk in chunks])
    y_all = np.concatenate([chunk[1] for chunk in chunks])
    
    # 80/20 split
    split = int(0.8 * len(X_all))
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]
    
    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}\n")
    
    results = {}
    
    # ===== SGD =====
    print("🔄 Training SGD Classifier...")
    memory_tracker.snapshot('sgd_batch_start')
    start = time.time()
    sgd = SGDClassifier(loss="log_loss", max_iter=20, random_state=42, n_jobs=-1)
    sgd.fit(X_train, y_train)
    sgd_pred = sgd.predict(X_test)
    sgd_time = time.time() - start
    sgd_metrics = compute_metrics(y_test, sgd_pred)
    memory_tracker.snapshot('sgd_batch_end')
    results['SGD'] = {**sgd_metrics, 'time': sgd_time}
    
    # ===== Passive-Aggressive =====
    print("🔄 Training Passive-Aggressive Classifier...")
    memory_tracker.snapshot('pa_batch_start')
    start = time.time()
    pa = PassiveAggressiveClassifier(C=0.01, max_iter=20, random_state=42, n_jobs=-1)
    pa.fit(X_train, y_train)
    pa_pred = pa.predict(X_test)
    pa_time = time.time() - start
    pa_metrics = compute_metrics(y_test, pa_pred)
    memory_tracker.snapshot('pa_batch_end')
    results['PA'] = {**pa_metrics, 'time': pa_time}
    
    # ===== XGBoost (Full Power) =====
    print("🔄 Training XGBoost (50 rounds, optimized)...")
    memory_tracker.snapshot('xgb_batch_start')
    start = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_model = xgb.train(
        {
            "objective": "multi:softmax",
            "num_class": 7,
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
            "nthread": -1
        },
        dtrain,
        num_boost_round=50
    )
    xgb_pred = xgb_model.predict(dtest)
    xgb_time = time.time() - start
    xgb_metrics = compute_metrics(y_test, xgb_pred)
    memory_tracker.snapshot('xgb_batch_end')
    results['XGBoost'] = {**xgb_metrics, 'time': xgb_time}
    
    # Print results
    print("\n📊 RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Time':<10}")
    print("-"*80)
    
    for model, metrics in results.items():
        print(f"{model:<15} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['time']:<10.2f}s")
    
    # Determine winner
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Winner: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
    
    return pd.DataFrame(results).T


def plot_results(streaming_results, output_dir='benchmark_results'):
    """Create visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Streaming Benchmark: Comprehensive Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy over chunks
    ax = axes[0, 0]
    ax.plot(streaming_results['chunk'], streaming_results['sgd_acc'], 
            'o-', label='SGD', linewidth=2, markersize=6)
    ax.plot(streaming_results['chunk'], streaming_results['pa_acc'], 
            's-', label='PA', linewidth=2, markersize=6)
    ax.plot(streaming_results['chunk'], streaming_results['xgb_acc'], 
            '^-', label='XGBoost', linewidth=2, markersize=6)
    ax.set_xlabel('Chunk', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Accuracy Evolution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Time comparison
    ax = axes[0, 1]
    models = ['SGD', 'PA', 'XGBoost']
    times = [
        streaming_results['sgd_time'].mean(),
        streaming_results['pa_time'].mean(),
        streaming_results['xgb_time'].mean()
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(models, times, color=colors, alpha=0.7)
    ax.set_ylabel('Average Time (seconds)', fontweight='bold')
    ax.set_title('Processing Speed per Chunk', fontweight='bold')
    ax.set_yscale('log')
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # 3. F1 Score comparison
    ax = axes[1, 0]
    ax.plot(streaming_results['chunk'], streaming_results['sgd_f1'], 
            'o-', label='SGD', linewidth=2, markersize=6)
    ax.plot(streaming_results['chunk'], streaming_results['pa_f1'], 
            's-', label='PA', linewidth=2, markersize=6)
    ax.plot(streaming_results['chunk'], streaming_results['xgb_f1'], 
            '^-', label='XGBoost', linewidth=2, markersize=6)
    ax.set_xlabel('Chunk', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('F1 Score Evolution', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Cumulative time
    ax = axes[1, 1]
    ax.plot(streaming_results['chunk'], streaming_results['sgd_time'].cumsum(), 
            'o-', label='SGD', linewidth=2, markersize=6)
    ax.plot(streaming_results['chunk'], streaming_results['pa_time'].cumsum(), 
            's-', label='PA', linewidth=2, markersize=6)
    ax.plot(streaming_results['chunk'], streaming_results['xgb_time'].cumsum(), 
            '^-', label='XGBoost', linewidth=2, markersize=6)
    ax.set_xlabel('Chunk', fontweight='bold')
    ax.set_ylabel('Cumulative Time (seconds)', fontweight='bold')
    ax.set_title('Total Processing Time', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to {output_dir}/benchmark_comparison.png")


def generate_report(streaming_results, batch_results, memory_df, output_dir='benchmark_results'):
    """Generate comprehensive markdown report"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/BENCHMARK_REPORT.md', 'w') as f:
        f.write("# 🔬 Machine Learning Benchmark Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now()}\n\n")
        
        f.write("## 📊 Executive Summary\n\n")
        f.write("This benchmark compares online learning (SGD, Passive-Aggressive) vs batch learning (XGBoost) ")
        f.write("across different scenarios to provide a fair, comprehensive evaluation.\n\n")
        
        f.write("## 🔄 Scenario 1: Real-Time Streaming\n\n")
        f.write("### Average Performance\n\n")
        f.write("| Model | Accuracy | F1 Score | Time/Chunk |\n")
        f.write("|-------|----------|----------|------------|\n")
        
        for model in ['sgd', 'pa', 'xgb']:
            acc = streaming_results[f'{model}_acc'].mean()
            f1 = streaming_results[f'{model}_f1'].mean()
            time = streaming_results[f'{model}_time'].mean()
            f.write(f"| {model.upper()} | {acc:.4f} | {f1:.4f} | {time:.4f}s |\n")
        
        f.write("\n### Key Insights\n\n")
        
        # Speed comparison
        pa_time = streaming_results['pa_time'].mean()
        xgb_time = streaming_results['xgb_time'].mean()
        speedup = xgb_time / pa_time
        f.write(f"- ⚡ **Speed**: Online learning is **{speedup:.1f}x faster** than XGBoost\n")
        
        # Accuracy comparison
        pa_acc = streaming_results['pa_acc'].mean()
        xgb_acc = streaming_results['xgb_acc'].mean()
        acc_diff = (xgb_acc - pa_acc) * 100
        if acc_diff > 0:
            f.write(f"- 🎯 **Accuracy**: XGBoost is {acc_diff:.2f}% more accurate\n")
        else:
            f.write(f"- 🎯 **Accuracy**: Online learning matches XGBoost performance\n")
        
        f.write("\n## 📦 Scenario 2: Batch Learning\n\n")
        f.write("### Performance\n\n")
        f.write(batch_results.to_markdown())
        
        f.write("\n\n## 🎓 Conclusions\n\n")
        f.write("### When to Use Online Learning (SGD/PA)\n")
        f.write("- ✅ Real-time predictions required\n")
        f.write("- ✅ Limited memory/compute\n")
        f.write("- ✅ Data streams continuously\n")
        f.write("- ✅ Concept drift expected\n\n")
        
        f.write("### When to Use Batch Learning (XGBoost)\n")
        f.write("- ✅ Maximum accuracy critical\n")
        f.write("- ✅ Full dataset available upfront\n")
        f.write("- ✅ Training time not critical\n")
        f.write("- ✅ Stable data distribution\n\n")
        
        f.write("### Fairness Assessment\n")
        f.write("- ⚖️ Both approaches tested in their optimal scenarios\n")
        f.write("- ⚖️ Same train/test splits used\n")
        f.write("- ⚖️ Statistical significance tested\n")
        f.write("- ⚖️ Multiple metrics reported\n")
    
    print(f"📄 Report saved to {output_dir}/BENCHMARK_REPORT.md")


def main():
    print("="*80)
    print("🔬 ADVANCED MACHINE LEARNING BENCHMARK")
    print("="*80)
    print("Comprehensive fairness testing with multiple scenarios and metrics\n")
    
    # Initialize memory tracker
    memory_tracker = MemoryTracker()
    
    # Load data
    chunks, all_classes = load_data(n_chunks=10, chunk_size=10000)
    
    # Run scenarios
    streaming_results = scenario_1_streaming(chunks, all_classes, memory_tracker)
    batch_results = scenario_2_batch(chunks, all_classes, memory_tracker)
    
    # Visualizations
    plot_results(streaming_results)
    
    # Memory analysis
    memory_df = memory_tracker.get_usage()
    
    # Generate report
    generate_report(streaming_results, batch_results, memory_df)
    
    # Save raw results
    streaming_results.to_csv('benchmark_results/streaming_results.csv', index=False)
    batch_results.to_csv('benchmark_results/batch_results.csv')
    memory_df.to_csv('benchmark_results/memory_usage.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 benchmark_results/benchmark_comparison.png")
    print("  📄 benchmark_results/BENCHMARK_REPORT.md")
    print("  📈 benchmark_results/streaming_results.csv")
    print("  📈 benchmark_results/batch_results.csv")
    print("  💾 benchmark_results/memory_usage.csv")
    print("\n🎯 Check BENCHMARK_REPORT.md for comprehensive analysis!")


if __name__ == "__main__":
    main()
