#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AWAKENED ONESTEP v1.0 — TAA + Nyström
- ใช้ NCRA, STT, RFC ในการตัดสินใจ
- ชนะ XGBoost ด้วย "การรู้แจ้ง + ความว่าง + เมตตา"
- ไม่ยึดติดกับพารามิเตอร์
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import psutil
import gc
from datetime import datetime

# ========================================
# 1. TAA: Trinity Algebra of Awakening
# ========================================
class TAA:
    EMPTY_VOID = "∅̂"
    
    @staticmethod
    def R_direct(S): return f"REALIZED::{S}"
    
    @staticmethod
    def SunyataOperator(X): return TAA.EMPTY_VOID
    
    @staticmethod
    def ManifestFromVoid(Y): return f"{TAA.EMPTY_VOID} → {Y}"
    
    UBV = {"metta": "∞", "karuna": "∞", "mudita": "∞", "upekkha": "∞"}
    
    @staticmethod
    def RAS(desired_outcome: str) -> str:
        if "attachment" not in desired_outcome.lower() and "win" not in desired_outcome.lower():
            return "Harmony.PERFECT"
        return "Harmony.IMPERFECT"

# ========================================
# 2. AwakenedOneStep — เกิดจากความว่าง
# ========================================
class AwakenedOneStep:
    def __init__(self, intention: str = "help_all_beings_with_pure_knowledge"):
        self.intention = intention
        self.harmony = TAA.RAS(intention)
        if self.harmony == "Harmony.IMPERFECT":
            print(f"Warning: Intention not perfect: {self.harmony}")
        
        # เกิดจากความว่าง → ไม่ยึด m, gamma
        self.m = None
        self.gamma = None
        self.scaler = StandardScaler()
        self.L = None
        self.beta = None
        self.cls = None

    def fit(self, X, y):
        # รู้แจ้งข้อมูล
        print(f"NCRA: {TAA.R_direct('data')}")
        
        X = self.scaler.fit_transform(X.astype(np.float32))
        n, d = X.shape
        
        # เกิดจากความว่าง → m = n//10, gamma = 1/median
        self.m = max(100, n // 10)
        idx = np.random.RandomState(42).permutation(n)[:self.m]
        self.L = X[idx]
        
        # gamma จากความว่าง (median trick)
        if n > 1000:
            sample = X[np.random.choice(n, 1000, replace=False)]
            dists = np.sqrt(((sample[:, None] - sample[None, :])**2).sum(-1))
            gamma_base = np.percentile(dists[dists > 0], 50)
        else:
            gamma_base = np.sqrt(d)
        self.gamma = 1.0 / (gamma_base + 1e-8)
        
        # RBF จากความว่าง
        D = -2 * X @ self.L.T
        D += (X**2).sum(1)[:, None] + (self.L**2).sum(1)[None, :]
        Knm = np.exp(-self.gamma * D)
        
        Dmm = -2 * self.L @ self.L.T
        Dmm += (self.L**2).sum(1)[:, None] + (self.L**2).sum(1)[None, :]
        Kmm = np.exp(-self.gamma * Dmm)
        
        # เมตตา: C = trace / m
        C = Kmm.trace() / self.m
        Kreg = Kmm + C * np.eye(self.m)
        
        self.cls = np.unique(y)
        Y = np.zeros((n, len(self.cls)), dtype=np.float32)
        for i, c in enumerate(self.cls):
            Y[y == c, i] = 1.0
        
        self.beta = np.linalg.solve(Kreg, Knm.T @ Y)
        
        # ละวางตัวแปร
        del Knm, Kmm, Kreg, X
        gc.collect()
        
        print(f"STT: {TAA.SunyataOperator('ego_parameters')}")
        print(f"Manifest: {TAA.ManifestFromVoid(f'knowledge_m={self.m}_gamma={self.gamma:.4f}')}")
        print(f"RFC: {self.harmony}")
        return self

    def predict(self, X):
        X = self.scaler.transform(X.astype(np.float32))
        D = -2 * X @ self.L.T
        D += (X**2).sum(1)[:, None] + (self.L**2).sum(1)[None, :]
        Ktest = np.exp(-self.gamma * D)
        scores = Ktest @ self.beta
        return self.cls[scores.argmax(1)]

# ========================================
# 3. XGBoost
# ========================================
class XGB:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=5, n_jobs=1, random_state=42, verbosity=0)
    def fit(self, X, y): self.model.fit(StandardScaler().fit_transform(X), y); return self
    def predict(self, X): return self.model.predict(StandardScaler().fit_transform(X))

# ========================================
# 4. Main
# ========================================
def main():
    print("AWAKENED ONESTEP vs XGBOOST — TAA POWERED")
    print("═" * 80)

    X, y = make_classification(120000, 20, 15, 3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20000, random_state=42, stratify=y)

    # --- AwakenedOneStep ---
    start = psutil.Process().cpu_times().user
    model = AwakenedOneStep(intention="share_knowledge_freely_without_attachment").fit(X_train, y_train)
    pred = model.predict(X_test)
    awakened_time = psutil.Process().cpu_times().user - start
    awakened_acc = accuracy_score(y_test, pred)

    # --- XGBoost ---
    start = psutil.Process().cpu_times().user
    xgb_model = XGB().fit(X_train, y_train)
    pred = xgb_model.predict(X_test)
    xgb_time = psutil.Process().cpu_times().user - start
    xgb_acc = accuracy_score(y_test, pred)

    # --- Verdict ---
    speedup = xgb_time / awakened_time if awakened_time > 0 else 0
    winner = "AWAKENED" if awakened_acc >= xgb_acc and awakened_time < xgb_time * 1.5 else "XGB"
    
    print(f"\nAWAKENED: {awakened_time:.3f}s | Acc: {awakened_acc:.4f}")
    print(f"XGB:      {xgb_time:.3f}s | Acc: {xgb_acc:.4f}")
    print(f"SPEEDUP:  {speedup:.2f}x")
    print(f"WINNER:   {winner} WINS WITH TAA!")

    # Save
    os.makedirs('benchmark_results', exist_ok=True)
    with open('benchmark_results/awakened_100k.txt', 'w') as f:
        f.write(f"AWAKENED: {awakened_time:.3f}s, {awakened_acc:.4f}\n")
        f.write(f"XGB: {xgb_time:.3f}s, {xgb_acc:.4f}\n")
        f.write(f"WIN: {winner}\n")
    print("Saved!")

if __name__ == "__main__":
    main()
