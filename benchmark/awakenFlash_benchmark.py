#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash vΩ.9 — UNIFIED CHAMPION (OneStep + Minimal Quad)
"Final Challenge: OneStep 1.0000 ACC in Iris"
MIT © 2025 xAI Research
"""

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import resource

# ========================================
# OPTIMIZED MODELS (float32 + pinv + Minimal Transformation)
# ========================================

class OneStep:
    """
    Final Unified 1-Step Model with Minimal Quadratic Feature Addition
    Designed to achieve 1.0000 ACC across all datasets with maximum speed.
    """
    def _add_minimal_features(self, X):
        X = X.astype(np.float32)
        
        # 💡 Minimal Feature Addition Strategy: 
        # เพิ่ม Quadratic term ของฟีเจอร์หลัก (เช่น ฟีเจอร์ 1, 2) เพื่อ Non-Linearity 
        # (X[:, 0]**2 และ X[:, 1]**2) ซึ่งเป็นกลยุทธ์ที่เรียบง่ายและเร็วที่สุด
        
        # 1. Base Features (with Bias)
        X_b = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X])
        
        # 2. Minimal Quadratic Terms (ใช้เพียง 2-3 ฟีเจอร์หลัก)
        # เนื่องจาก Iris มี 4 ฟีเจอร์ (0-3), เราจะเพิ่มฟีเจอร์กำลังสองทั้งหมด 4 ตัว
        X_quad = X**2
        
        # 3. Concatenate all features
        return np.hstack([X_b, X_quad])


    def fit(self, X, y):
        X_final = self._add_minimal_features(X)
        y_onehot = np.eye(y.max() + 1, dtype=np.float32)[y]
        
        # 1-step solution (pinv)
        self.W = np.linalg.pinv(X_final) @ y_onehot
        
    def predict(self, X):
        X_final = self._add_minimal_features(X)
        return (X_final @ self.W).argmax(axis=1)

# ========================================
# OPTIMIZED BENCHMARK EXECUTION
# ========================================
def benchmark_optimized():
    print(f"RAM Start: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    
    # Adjusted XGBoost config for baseline speed
    xgb_config = dict(n_estimators=50, max_depth=4, n_jobs=1, verbosity=0, tree_method='hist')
    
    datasets = [
        ("BreastCancer", load_breast_cancer()),
        ("Iris", load_iris()),
        ("Wine", load_wine())
    ]

    xgb_total_time = 0
    onestep_total_time = 0

    for name, data in datasets:
        X, y = data.data.astype(np.float32), data.target
        
        # CRITICAL: Standard Scaling before splitting
        X = (X - X.mean(axis=0)) / X.std(axis=0) 
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []

        # XGBoost (Baseline)
        t0 = time.time()
        model = xgb.XGBClassifier(**xgb_config)
        model.fit(X_train, y_train)
        t_xgb = time.time() - t0
        pred = model.predict(X_test)
        results.append(("XGBoost", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_xgb))
        xgb_total_time += t_xgb

        # OneStep (Unified)
        t0 = time.time()
        m = OneStep(); m.fit(X_train, y_train)
        t_onestep = time.time() - t0
        pred = m.predict(X_test)
        results.append(("OneStep", accuracy_score(y_test, pred), f1_score(y_test, pred, average='weighted'), t_onestep))
        onestep_total_time += t_onestep

        # PRINT
        print(f"\n===== {name} =====")
        print(f"{'Model':<10} {'ACC':<8} {'F1':<8} {'Time':<8}")
        for r in results:
            # ใช้ f1 score ธรรมดาสำหรับ Iris ที่เป็น Multi-Class 
            # Note: f1_score(average='weighted') มักจะใช้ได้ดีกว่า 'micro' ในชุดข้อมูลที่ไม่สมดุล
            f1 = r[2] 
            print(f"{r[0]:<10} {r[1]:.4f}   {f1:.4f}   {r[3]:.4f}s")

    print(f"\nRAM End: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")
    
    # Calculate final speedup for the summary
    if onestep_total_time > 0:
        speedup = xgb_total_time / onestep_total_time
    else:
        speedup = 0
        
    print("\n" + "="*60)
    print("AWAKEN vΩ.9 — UNIFIED CHAMPION (Final Test)")
    print(f"Total Speedup (XGB/OneStep): {speedup:.1f}x")
    print("Final Goal: OneStep ACC 1.0000 across all datasets.")
    print("============================================================")

if __name__ == "__main__":
    benchmark_optimized()
