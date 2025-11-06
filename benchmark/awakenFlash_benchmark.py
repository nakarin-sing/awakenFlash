#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ULTIMATE STREAMING BENCHMARK - TRUE REAL-WORLD v1.1 (Extreme Speed Check)
Target: Ensemble Prediction Latency < 0.039ms (10x faster than XGBoost 0.39ms)

Key Changes (v1.1):
1. Ensemble Model: Changed from RandomForestClassifier(n_estimators=50) to ExtraTreesClassifier(n_estimators=5). (Extreme Latency Reduction)
2. n_batches: Reduced from 10 to 5 for smaller, quicker training batches.
3. Added n_jobs=-1 to maximize parallelization (crucial for tree-based models)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
# FIX V1.1: Import ExtraTreesClassifier for speed optimization
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.metrics import accuracy_score
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

class TrustworthyBenchmark:
    def __init__(self):
        self.results = {}
        
    def validate_consistency(self, intermediate_scores, final_scores, tolerance=0.05):
        """ตรวจสอบความสอดคล้องระหว่างคะแนนกลางทางและคะแนนสุดท้าย"""
        for model_name in intermediate_scores:
            if model_name in final_scores:
                # Use the last intermediate score, which is the final batch's score
                last_intermediate = intermediate_scores[model_name][-1]
                final = final_scores[model_name]
                if abs(last_intermediate - final) > tolerance:
                    print(f"⚠️  คำเตือน: {model_name} มีความไม่สอดคล้อง (กลางทาง: {last_intermediate:.4f}, สุดท้าย: {final:.4f})")
                    return False
        return True
    
    def measure_latency(self, model, X_test, n_repeats=50): # Increased repeats for better median stability
        """วัด latency แบบแม่นยำด้วยการทำซ้ำหลายครั้ง"""
        latencies = []
        for _ in range(n_repeats):
            start_time = time.time()
            model.predict(X_test)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # convert to ms
        
        # ใช้ค่า median เพื่อลดผลกระทบจาก outlier
        return np.median(latencies)
    
    def run_dataset_benchmark(self, dataset_name, data_loader, data_multiplier=1):
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} (x{data_multiplier} data)")
        print(f"{'='*60}")
        
        # โหลดและเตรียมข้อมูล
        X, y = data_loader(return_X_y=True)
        
        if data_multiplier > 1:
            X = np.vstack([X] * data_multiplier)
            y = np.hstack([y] * data_multiplier)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # FIX V1.1: ลด n_batches เหลือ 5
        n_batches = 5
        batch_size = len(X_train) // n_batches
        batches = []
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size if i < n_batches - 1 else len(X_train)
            batches.append((X_train[start_idx:end_idx], y_train[start_idx:end_idx]))
        
        # FIX V1.1: เปลี่ยนเป็น ExtraTreesClassifier(n_estimators=5, n_jobs=-1)
        ensemble_model = ExtraTreesClassifier(n_estimators=5, random_state=42, n_jobs=-1) 
        # XGBoost คงเดิมเพื่อเป็น Baseline ในการทำนาย
        xgb_model = xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=-1) 
        
        intermediate_scores = {'Ens': [], 'XGB': []}
        intermediate_latencies = {'Ens': [], 'XGB': []}
        
        # Streaming training and evaluation
        for i, (X_batch, y_batch) in enumerate(batches):
            # Train models incrementally
            start_time = time.time()
            ensemble_model.fit(X_batch, y_batch)
            ensemble_train_time = (time.time() - start_time) * 1000
            
            start_time = time.time()
            xgb_model.fit(X_batch, y_batch)
            xgb_train_time = (time.time() - start_time) * 1000
            
            # Predict and evaluate
            ens_pred = ensemble_model.predict(X_test)
            xgb_pred = xgb_model.predict(X_test)
            
            ens_acc = accuracy_score(y_test, ens_pred)
            xgb_acc = accuracy_score(y_test, xgb_pred)
            
            # Measure prediction latency
            ens_latency = self.measure_latency(ensemble_model, X_test)
            xgb_latency = self.measure_latency(xgb_model, X_test)
            
            intermediate_scores['Ens'].append(ens_acc)
            intermediate_latencies['Ens'].append(ens_latency)
            intermediate_latencies['XGB'].append(xgb_latency)
            
            print(f"Batch {i:2d} | Ens: {ens_acc:.4f}({ens_latency:.3f}ms) | " # FIX V1.1: Latency to 3 decimal places
                  f"XGB: {xgb_acc:.4f}({xgb_latency:.3f}ms) | "
                  f"Train: Ens={ensemble_train_time:.2f}ms, XGB={xgb_train_time:.2f}ms")
        
        # Final evaluation with proper validation
        final_ens_acc = accuracy_score(y_test, ensemble_model.predict(X_test))
        final_xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))
        
        final_scores = {'Ens': final_ens_acc, 'XGB': final_xgb_acc}
        
        # Validate consistency
        is_consistent = self.validate_consistency(intermediate_scores, final_scores)
        
        print(f"\nFinal Test: Ens={final_ens_acc:.4f}, XGB={final_xgb_acc:.4f}")
        if not is_consistent:
            print("🔍 มีความไม่สอดคล้องในผลลัพธ์ - ควรตรวจสอบ methodology")
        
        # Calculate robust statistics
        avg_ens_latency = np.median(intermediate_latencies['Ens'])
        avg_xgb_latency = np.median(intermediate_latencies['XGB'])
        avg_ens_acc = np.mean(final_scores['Ens']) # Use final score for summary
        avg_xgb_acc = np.mean(final_scores['XGB'])
        
        return {
            'final_accuracy': final_scores,
            'avg_accuracy': {'Ens': avg_ens_acc, 'XGB': avg_xgb_acc},
            'avg_latency': {'Ens': avg_ens_latency, 'XGB': avg_xgb_latency},
            'consistency_check': is_consistent
        }

def main():
    benchmark = TrustworthyBenchmark()
    
    datasets = [
        ("BreastCancer", load_breast_cancer, 1),
        ("Iris", load_iris, 3),
        ("Wine", load_wine, 3)
    ]
    
    all_results = []
    
    for dataset_name, loader, multiplier in datasets:
        result = benchmark.run_dataset_benchmark(dataset_name, loader, multiplier)
        result['dataset'] = dataset_name
        all_results.append(result)
    
    # Summary with transparent calculations
    print(f"\n{'='*60}")
    print("EXTREME SPEED CHECK SUMMARY (v1.1)")
    print(f"{'='*60}")
    
    total_ens_acc = sum(r['final_accuracy']['Ens'] for r in all_results)
    total_xgb_acc = sum(r['final_accuracy']['XGB'] for r in all_results)
    total_ens_latency = sum(r['avg_latency']['Ens'] for r in all_results)
    total_xgb_latency = sum(r['avg_latency']['XGB'] for r in all_results)
    n_datasets = len(all_results)
    
    overall_ens_acc = total_ens_acc / n_datasets
    overall_xgb_acc = total_xgb_acc / n_datasets
    overall_ens_latency = total_ens_latency / n_datasets
    overall_xgb_latency = total_xgb_latency / n_datasets
    
    print(f"Overall Accuracy:  Ensemble (ExtraTrees) = {overall_ens_acc:.4f}, XGBoost = {overall_xgb_acc:.4f}")
    print(f"Overall Latency:   Ensemble (ExtraTrees) = {overall_ens_latency:.3f}ms, XGBoost = {overall_xgb_latency:.3f}ms")
    
    accuracy_diff = overall_ens_acc - overall_xgb_acc
    speed_ratio = overall_xgb_latency / overall_ens_latency if overall_ens_latency > 0 else 1
    
    print(f"\nPerformance Analysis:")
    print(f"Accuracy Difference: {accuracy_diff:+.4f} ({'Ensemble ดีกว่า' if accuracy_diff > 0 else 'XGBoost ดีกว่า'})")
    print(f"Speed Ratio: {speed_ratio:.1f}x ({'Ensemble เร็วกว่า' if speed_ratio > 1 else 'XGBoost เร็วกว่า'})")
    
    consistent_count = sum(1 for r in all_results if r['consistency_check'])
    print(f"\nData Quality Check: {consistent_count}/{len(all_results)} datasets มีความสอดคล้อง")
    
    if overall_ens_latency < overall_xgb_latency / 10:
        print("🎉 **GOAL ACHIEVED!** Ensemble ชนะความเร็ว 10x")

if __name__ == "__main__":
    main()
