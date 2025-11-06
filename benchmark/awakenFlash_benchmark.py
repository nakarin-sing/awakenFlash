#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTRA STABLE BENCHMARK 10 CHUNKS - Enhanced for Victory
10 chunks with improved learning for consistent victory over XGBoost
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

class EnhancedFeatureEngine:
    """
    Enhanced Feature Engineering - More strategic features
    """
    
    def __init__(self, max_interactions=4):
        self.max_interactions = max_interactions
        self.interaction_pairs = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def fit_transform(self, X):
        """Create enhanced features"""
        X = self.scaler.fit_transform(X)
        n_features = X.shape[1]
        
        # Enhanced feature selection using multiple metrics
        variances = np.var(X, axis=0)
        # Use mean absolute deviation for robustness
        mad = np.mean(np.abs(X - np.mean(X, axis=0)), axis=0)
        
        # Combined importance score
        combined_importance = variances * (1 + 0.5 * mad)
        top_indices = np.argsort(combined_importance)[-6:]  # Top 6 features
        
        self.interaction_pairs = []
        for i in range(len(top_indices)):
            for j in range(i+1, min(i+4, len(top_indices))):
                if len(self.interaction_pairs) < self.max_interactions:
                    self.interaction_pairs.append((top_indices[i], top_indices[j]))
        
        # Create enhanced interaction features
        X_interactions = []
        for i, j in self.interaction_pairs:
            # Multiplication
            mult = (X[:, i] * X[:, j]).reshape(-1, 1)
            X_interactions.append(mult)
            
            # Sum (very stable)
            sum_feat = (X[:, i] + X[:, j]).reshape(-1, 1)
            X_interactions.append(sum_feat)
            
            # Difference (captures relationships)
            diff = (X[:, i] - X[:, j]).reshape(-1, 1)
            X_interactions.append(diff)
        
        if X_interactions:
            X_enhanced = np.hstack([X] + X_interactions)
            print(f"   Features: {X.shape[1]} → {X_enhanced.shape[1]} (enhanced)")
            return X_enhanced
        return X
    
    def transform(self, X):
        """Apply enhanced transformations"""
        X = self.scaler.transform(X)
        
        if self.interaction_pairs is None:
            return X
        
        X_interactions = []
        for i, j in self.interaction_pairs:
            mult = (X[:, i] * X[:, j]).reshape(-1, 1)
            X_interactions.append(mult)
            sum_feat = (X[:, i] + X[:, j]).reshape(-1, 1)
            X_interactions.append(sum_feat)
            diff = (X[:, i] - X[:, j]).reshape(-1, 1)
            X_interactions.append(diff)
        
        if X_interactions:
            return np.hstack([X] + X_interactions)
        return X


class EnhancedEnsemble:
    """
    Enhanced Ensemble - Optimized for 10 chunks
    """
    
    def __init__(self, memory_size=12000, feature_engine=None):
        self.models = []
        self.weights = np.ones(4) / 4
        self.all_data_X = []
        self.all_data_y = []
        self.memory_size = memory_size
        self.feature_engine = feature_engine
        self.performance_history = []
        self.learning_rates = [0.1, 0.05, 0.1, 0.08]  # Different learning rates
        
        # 4 diverse models with varied parameters
        self.models.append(SGDClassifier(
            loss='log_loss', 
            learning_rate='optimal', 
            max_iter=10,
            warm_start=True, 
            random_state=42, 
            alpha=0.0008,
            penalty='l2'
        ))
        
        self.models.append(PassiveAggressiveClassifier(
            C=0.08, 
            max_iter=10, 
            warm_start=True, 
            random_state=43,
            shuffle=True
        ))
        
        self.models.append(SGDClassifier(
            loss='modified_huber',
            learning_rate='constant',
            eta0=0.02,
            max_iter=10,
            warm_start=True,
            random_state=44,
            alpha=0.0006
        ))
        
        self.models.append(SGDClassifier(
            loss='hinge',
            learning_rate='optimal',
            max_iter=10,
            warm_start=True,
            random_state=45,
            alpha=0.001,
            penalty='l1'
        ))
        
        self.first_fit = True
        self.classes_ = None
        self.chunk_count = 0
    
    def _update_weights_strategic(self, X_val, y_val):
        """Strategic weight update based on performance and stability"""
        model_performances = []
        model_stabilities = []
        
        for i, model in enumerate(self.models):
            try:
                current_acc = model.score(X_val, y_val)
                model_performances.append(max(0.1, current_acc))
                
                # Calculate stability (consistency with recent performance)
                if len(self.performance_history) > 2 and i < len(self.performance_history[-1]):
                    recent_perf = self.performance_history[-1][i]
                    stability = 1.0 - min(1.0, abs(current_acc - recent_perf) / max(0.1, recent_perf))
                else:
                    stability = 0.7  # Default stability
                model_stabilities.append(stability)
            except:
                model_performances.append(0.1)
                model_stabilities.append(0.5)
        
        # Strategic weighting: performance × stability
        strategic_weights = np.array(model_performances) * np.array(model_stabilities)
        strategic_weights = np.maximum(0.1, strategic_weights)
        
        # Adaptive momentum based on overall stability
        avg_stability = np.mean(model_stabilities)
        momentum = 0.4 + (avg_stability * 0.3)  # 0.4-0.7 based on stability
        
        new_weights = momentum * self.weights + (1 - momentum) * strategic_weights
        
        # Normalize
        total = np.sum(new_weights)
        if total > 0:
            self.weights = new_weights / total
    
    def partial_fit(self, X, y, classes=None):
        """Enhanced online learning with strategic reinforcement"""
        if self.first_fit and classes is not None:
            self.classes_ = classes
            self.first_fit = False
        
        self.chunk_count += 1
        
        # Enhanced memory management - keep more recent data
        self.all_data_X.append(X)
        self.all_data_y.append(y)
        
        total_samples = sum(len(x) for x in self.all_data_X)
        while total_samples > self.memory_size and len(self.all_data_X) > 3:
            # Keep at least 3 recent chunks
            self.all_data_X.pop(0)
            self.all_data_y.pop(0)
            total_samples = sum(len(x) for x in self.all_data_X)
        
        # Train all models
        current_performances = []
        for model in self.models:
            try:
                if classes is not None:
                    model.partial_fit(X, y, classes=classes)
                else:
                    model.partial_fit(X, y)
                
                # Track current performance
                current_acc = model.score(X, y)
                current_performances.append(current_acc)
            except Exception as e:
                current_performances.append(0.1)
                continue
        
        self.performance_history.append(current_performances)
        if len(self.performance_history) > 5:
            self.performance_history.pop(0)
        
        # Strategic reinforcement learning (every 3 chunks)
        if len(self.all_data_X) >= 3 and self.chunk_count % 3 == 0:
            # Use recent 3 chunks for reinforcement
            recent_X = np.vstack(self.all_data_X[-3:])
            recent_y = np.concatenate(self.all_data_y[-3:])
            
            # Use larger sample for better learning
            n_samples = min(2000, len(recent_X))
            indices = np.random.choice(len(recent_X), n_samples, replace=False)
            X_sample = recent_X[indices]
            y_sample = recent_y[indices]
            
            # Reinforce top 3 models
            top_indices = np.argsort(self.weights)[-3:]
            for idx in top_indices:
                try:
                    self.models[idx].partial_fit(X_sample, y_sample)
                except:
                    pass
    
    def predict(self, X):
        """Enhanced prediction with strategic weighting"""
        if not self.models or self.classes_ is None:
            return np.zeros(len(X))
        
        all_predictions = []
        valid_weights = []
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                all_predictions.append(pred)
                valid_weights.append(self.weights[i])
            except:
                continue
        
        if not all_predictions:
            return np.zeros(len(X))
        
        # Strategic weighted voting
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()
        
        n_samples = len(X)
        n_classes = len(self.classes_)
        vote_matrix = np.zeros((n_samples, n_classes))
        
        for pred, weight in zip(all_predictions, valid_weights):
            for i, cls in enumerate(self.classes_):
                vote_matrix[:, i] += (pred == cls) * weight
        
        return self.classes_[np.argmax(vote_matrix, axis=1)]


def load_data_10_chunks():
    """Load data for 10 chunks"""
    print("📦 Loading dataset for 10 chunks...")
    
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        df = pd.read_csv(url, header=None, nrows=45000)  # More data for 10 chunks
    except:
        # Enhanced synthetic data
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=45000, n_features=54, n_informative=20,
            n_redundant=10, n_classes=7, random_state=42,
            n_clusters_per_class=1, flip_y=0.015
        )
        df = pd.DataFrame(X)
        df['target'] = y
    
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values
    
    if y_all.max() > 6:
        y_all = y_all % 7
    
    print(f"   Dataset: {X_all.shape}, Classes: {len(np.unique(y_all))}")
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)
    
    # 10 chunks with smaller size
    chunk_size = 3000
    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, min(len(X_all), 10 * chunk_size), chunk_size)]
    
    return chunks[:10], np.unique(y_all)


def enhanced_10_chunks_benchmark():
    """
    ENHANCED 10 CHUNKS BENCHMARK - Optimized for victory
    """
    print("\n" + "="*60)
    print("🏆 ENHANCED 10 CHUNKS BENCHMARK")
    print("="*60)
    print("Target: Consistent victory over XGBoost with 10 chunks\n")
    
    # Load data for 10 chunks
    chunks, all_classes = load_data_10_chunks()
    
    # Enhanced feature engine
    feature_engine = EnhancedFeatureEngine(max_interactions=4)
    
    # Initialize enhanced ensemble
    enhanced = EnhancedEnsemble(memory_size=12000, feature_engine=feature_engine)
    
    # Baseline models
    sgd = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        max_iter=8,
        warm_start=True,
        random_state=42
    )
    
    pa = PassiveAggressiveClassifier(
        C=0.1,
        max_iter=8,
        warm_start=True,
        random_state=42
    )
    
    # XGBoost with enhanced settings
    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 4  # Larger window for more stable learning
    
    # Initialize variables
    enhanced_acc = sgd_acc = pa_acc = xgb_acc = 0.0
    results = []
    
    # Fit feature engine
    if chunks and len(chunks) > 0:
        try:
            X_sample, _ = chunks[0]
            feature_engine.fit_transform(X_sample[:1000])
        except:
            print("   Feature engine fitting failed, using original features")
    
    print(f"Starting 10 chunks benchmark...")
    
    for chunk_id, (X_chunk, y_chunk) in enumerate(chunks, 1):
        # Safe splitting
        split = int(0.7 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]
        
        # Feature transformation
        try:
            X_train_eng = feature_engine.transform(X_train)
            X_test_eng = feature_engine.transform(X_test)
        except:
            X_train_eng, X_test_eng = X_train, X_test
        
        print(f"Chunk {chunk_id:2d}/10 | Train: {len(X_train)}, Test: {len(X_test)}")
        
        # ===== Enhanced Ensemble =====
        try:
            start = time.time()
            if chunk_id == 1:
                enhanced.partial_fit(X_train_eng, y_train, classes=all_classes)
            else:
                enhanced.partial_fit(X_train_eng, y_train)
            enhanced_pred = enhanced.predict(X_test_eng)
            enhanced_acc = accuracy_score(y_test, enhanced_pred)
            enhanced_time = time.time() - start
            
            # Update weights based on this performance
            enhanced._update_weights_strategic(X_test_eng, y_test)
        except Exception as e:
            enhanced_acc = 0.0
            enhanced_time = 0.0
        
        # ===== SGD =====
        try:
            start = time.time()
            if chunk_id == 1:
                sgd.partial_fit(X_train_eng, y_train, classes=all_classes)
            else:
                sgd.partial_fit(X_train_eng, y_train)
            sgd_pred = sgd.predict(X_test_eng)
            sgd_acc = accuracy_score(y_test, sgd_pred)
            sgd_time = time.time() - start
        except Exception as e:
            sgd_acc = 0.0
            sgd_time = 0.0
        
        # ===== PA =====
        try:
            start = time.time()
            if chunk_id == 1:
                pa.partial_fit(X_train_eng, y_train, classes=all_classes)
            else:
                pa.partial_fit(X_train_eng, y_train)
            pa_pred = pa.predict(X_test_eng)
            pa_acc = accuracy_score(y_test, pa_pred)
            pa_time = time.time() - start
        except Exception as e:
            pa_acc = 0.0
            pa_time = 0.0
        
        # ===== XGBoost =====
        try:
            start = time.time()
            xgb_all_X.append(X_train_eng)
            xgb_all_y.append(y_train)
            
            if len(xgb_all_X) > WINDOW_SIZE:
                xgb_all_X = xgb_all_X[-WINDOW_SIZE:]
                xgb_all_y = xgb_all_y[-WINDOW_SIZE:]
            
            X_xgb = np.vstack(xgb_all_X)
            y_xgb = np.concatenate(xgb_all_y)
            
            dtrain = xgb.DMatrix(X_xgb, label=y_xgb)
            dtest = xgb.DMatrix(X_test_eng, label=y_test)
            
            xgb_model = xgb.train(
                {
                    "objective": "multi:softmax",
                    "num_class": len(all_classes),
                    "max_depth": 4,
                    "eta": 0.1,
                    "subsample": 0.8,
                    "verbosity": 0,
                    "nthread": 1
                },
                dtrain,
                num_boost_round=10
            )
            
            xgb_pred = xgb_model.predict(dtest)
            xgb_acc = accuracy_score(y_test, xgb_pred)
            xgb_time = time.time() - start
        except Exception as e:
            xgb_acc = 0.0
            xgb_time = 0.0
        
        # Store results
        results.append({
            'chunk': chunk_id,
            'enhanced_acc': enhanced_acc,
            'sgd_acc': sgd_acc,
            'pa_acc': pa_acc,
            'xgb_acc': xgb_acc,
        })
        
        print(f"  Enhanced: {enhanced_acc:.3f} ({enhanced_time:.2f}s)")
        print(f"  SGD:      {sgd_acc:.3f} ({sgd_time:.2f}s)")
        print(f"  PA:       {pa_acc:.3f} ({pa_time:.2f}s)")
        print(f"  XGB:      {xgb_acc:.3f} ({xgb_time:.2f}s)")
    
    # Enhanced results analysis
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("📊 10 CHUNKS ENHANCED RESULTS")
        print("="*60)
        
        # Calculate comprehensive metrics
        accuracies = {}
        stabilities = {}
        improvements = []
        
        for model in ['enhanced', 'sgd', 'pa', 'xgb']:
            if f'{model}_acc' in df_results.columns:
                accs = df_results[f'{model}_acc'].values
                acc_mean = np.mean(accs)
                acc_std = np.std(accs)
                stability = 1.0 - (acc_std / max(0.1, acc_mean))
                
                accuracies[model] = acc_mean
                stabilities[model] = stability
                
                print(f"{model.upper():10s}: {acc_mean:.4f} ± {acc_std:.4f} (stab: {stability:.3f})")
                
                # Track improvement over chunks
                if len(accs) >= 3:
                    early_avg = np.mean(accs[:3])
                    late_avg = np.mean(accs[-3:])
                    improvement = late_avg - early_avg
                    improvements.append(improvement)
        
        # Determine winner considering both accuracy and stability
        def comprehensive_score(acc, stability, improvement):
            return acc * 0.6 + stability * 0.3 + improvement * 0.1
        
        model_scores = {}
        for model in accuracies.keys():
            imp = improvements[list(accuracies.keys()).index(model)] if improvements else 0
            model_scores[model] = comprehensive_score(accuracies[model], stabilities[model], imp)
        
        winner = max(model_scores, key=model_scores.get)
        enhanced_acc = accuracies.get('enhanced', 0.0)
        xgb_acc = accuracies.get('xgb', 0.0)
        margin = (enhanced_acc - xgb_acc) * 100
        
        print(f"\n🏆 COMPREHENSIVE WINNER: {winner.upper()} (score: {model_scores[winner]:.4f})")
        print(f"📈 Accuracy Margin: Enhanced {margin:+.2f}% over XGBoost")
        print(f"🎯 Stability: Enhanced {stabilities.get('enhanced', 0):.3f} vs XGB {stabilities.get('xgb', 0):.3f}")
        
        # Victory analysis
        if winner == 'enhanced' and margin > 1.0:
            print("🎉 DOMINANT VICTORY: Enhanced ensemble clearly outperforms XGBoost!")
        elif winner == 'enhanced' and margin > 0.5:
            print("✅ SOLID VICTORY: Enhanced ensemble beats XGBoost consistently!")
        elif winner == 'enhanced':
            print("⚠️  NARROW VICTORY: Enhanced ensemble edges out XGBoost")
        else:
            # Calculate improvement from previous benchmark
            previous_ultra = 0.6618
            improvement = (enhanced_acc - previous_ultra) * 100
            print(f"🔁 XGBoost wins, but Enhanced improved by {improvement:+.2f}% over Ultra")
        
        # Learning analysis
        print(f"\n📊 LEARNING ANALYSIS:")
        if len(df_results) >= 5:
            first_half = df_results['enhanced_acc'].iloc[:5].mean()
            second_half = df_results['enhanced_acc'].iloc[5:].mean()
            learning_gain = (second_half - first_half) * 100
            print(f"   Enhanced learning gain: {learning_gain:+.2f}% (first 5 vs last 5 chunks)")
        
        # Save detailed results
        try:
            os.makedirs('benchmark_results', exist_ok=True)
            df_results.to_csv('benchmark_results/enhanced_10_chunks_results.csv', index=False)
            print("💾 Results saved to benchmark_results/enhanced_10_chunks_results.csv")
        except:
            print("💾 Could not save results")
        
        return True, accuracies, stabilities
    else:
        print("❌ No results generated")
        return False, {}, {}


def main():
    """Main function for 10 chunks benchmark"""
    print("="*60)
    print("🏆 ENHANCED 10 CHUNKS ML BENCHMARK")
    print("="*60)
    print("Mission: Victory over XGBoost with 10 chunks learning\n")
    
    start_time = time.time()
    success = False
    accuracies = {}
    stabilities = {}
    
    try:
        success, accuracies, stabilities = enhanced_10_chunks_benchmark()
        total_time = time.time() - start_time
        
        print(f"\n✅ BENCHMARK COMPLETED in {total_time:.1f}s")
        
        if success:
            if 'enhanced' in accuracies and 'xgb' in accuracies:
                margin = (accuracies['enhanced'] - accuracies['xgb']) * 100
                if margin > 0:
                    print(f"🎯 MISSION ACCOMPLISHED: Enhanced wins by {margin:.2f}%!")
                else:
                    print(f"📊 Competitive results: Margin = {margin:.2f}%")
            
            if total_time < 8:
                print("⚡ Excellent: Fast execution with 10 chunks")
            elif total_time < 15:
                print("⏱️  Good: Reasonable time for 10 chunks")
                
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            os.makedirs('benchmark_results', exist_ok=True)
            with open('benchmark_results/error_10_chunks.log', 'w') as f:
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            pass
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
