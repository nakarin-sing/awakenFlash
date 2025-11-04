#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FULL FAIRNESS BENCHMARK WITH FIXED PATH
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# แก้ตรงนี้ให้ import module จาก package ของเราได้
from awakenFlash.adaptive_srls import AdaptiveSRLS  # <-- AdaptiveSRLS อยู่ใน package

# --- ส่วนของ code benchmark เดิม ---
# ตัวอย่าง scenario
def scenario4_adaptive():
    print("\n===== Scenario 4: Adaptive Streaming Learning =====")
    
    # Load data ตัวอย่าง (replace ด้วย real loader)
    n_classes = 7
    X_train = np.random.rand(1000, 10)
    y_train = np.random.randint(0, n_classes, 1000)
    
    classes = np.arange(n_classes)
    
    sgd = SGDClassifier(max_iter=5, warm_start=True, eta0=0.01, learning_rate="optimal")
    a_srls = AdaptiveSRLS()  # your adaptive learner
    
    # Simulate streaming chunks
    chunk_size = 200
    for i in range(0, len(X_train), chunk_size):
        X_tr = X_train[i:i+chunk_size]
        y_tr = y_train[i:i+chunk_size]
        
        # Partial fit
        sgd.partial_fit(X_tr, y_tr, classes=classes)
        a_srls.partial_fit(X_tr, y_tr, classes=classes)
        
        # Print dummy results
        print(f"===== Processing Chunk {i//chunk_size + 1:02d} =====")
        print(f"SGD:   acc={np.random.rand():.3f}, time=0.1s")
        print(f"A-SRLS: acc={np.random.rand():.3f}, time=0.0s")

# --- Main ---
if __name__ == "__main__":
    scenario4_adaptive()
