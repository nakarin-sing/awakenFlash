#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash core — train + infer functions (INT8)
"""

import numpy as np
from numba import njit, prange

# --------------------------
# Global Config (UPDATED FOR SPEED VALIDATION N_F=40)
# --------------------------
N_SAMPLES = 100_000 # 100k samples for better training
N_FEATURES = 40     # Target for speed re-validation
N_CLASSES = 3       # 3 classes (เพื่อให้เหมือน benchmark v1.5)
B = 1024
H = 448             # Hidden layer size, adjusted for N_SAMPLES
CONF_THRESHOLD = 80
LS = 0.006          # Label Smoothing factor (0.6% noise)

# --------------------------
# Core Train Function (FIXED LOGIC)
# --------------------------
@njit(parallel=True, fastmath=True, cache=True)
def train_step(X_i8, y, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS):
    n = X_i8.shape[0]
    num_classes = len(b2)

    # Note: เราไม่มี lut_exp ใน core file นี้ เราจะใช้ค่า d แทน exp(d) ใน Backprop
    for i in prange(n):
        x = X_i8[i]
        target = y[i]
        h = np.zeros(H, np.int64)
        
        # Forward Pass - Layer 1 (W1 * X)
        for j in range(H):
            acc = b1[j]
            for k in range(indptr[j], indptr[j+1]):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0) # ReLU Approximation
            
        # Forward Pass - Layer 2 (W2 * H) -> Logits
        logits = b2.copy()
        for j in range(H):
            if h[j]:
                for c in range(num_classes):
                    logits[c] += h[j] * W2[j, c]
        
        # Softmax / Confidence Approximation (ใช้ค่า d แทน e^d)
        min_l = logits.min()
        sum_d = 0 # ใช้ sum_d แทน sum_e 
        max_l = logits[0]
        for c in range(num_classes):
            d = min(127, max(0, (logits[c] - min_l) >> 1))
            sum_d += d
            if logits[c] > max_l:
                max_l = logits[c]
        conf = (max_l * 100) // max(sum_d, 1)
        pred = np.argmax(logits)
        
        # Early Exit / Target Label
        if conf < CONF_THRESHOLD:
            final_target = target
        else:
            final_target = pred
            
        # Target Label Smoothing
        tgt = np.zeros(num_classes, np.int64)
        tgt[final_target] = 127
        tgt = ((1 - LS) * tgt + LS * 127 // num_classes).astype(np.int64)
        
        # Probability Vector (ใช้ค่า d แทน e^d)
        prob = np.zeros(num_classes, np.int64)
        sum_d_norm = 0
        for c in range(num_classes):
            # BUG FIX: ใน core เดิมใช้ min_l ผิดพลาด เราจะใช้ d_value ที่ถูก scale แล้ว
            d_value = min(127, max(0, (logits[c] - min_l) >> 1))
            prob[c] = d_value
            sum_d_norm += d_value # sum_d_norm
            
        # Normalization
        prob = (prob * 127) // max(sum_d_norm, 1)
        
        # Error Signal (dL) - FIXED SCALING
        # เราใช้ dL = prob - tgt โดยไม่หาร 127/ns เหมือน v1.4/v1.5
        dL = np.clip(prob - tgt, -5, 5) 
        
        # Backpropagation Update
        for c in range(num_classes):
            # Update b2 and W2
            db = dL[c] // 20 # Fixed Learning Rate scaling
            b2[c] -= db
            for j in range(H):
                if h[j]:
                    W2[j, c] -= (h[j] * db) // 127
                    
        for j in range(H):
            # Update b1 and W1
            dh = 0
            for c in range(num_classes):
                if h[j]:
                    dh += dL[c] * W2[j, c]
            db1 = dh // 20 # Fixed Learning Rate scaling
            b1[j] -= db1
            for k in range(indptr[j], indptr[j+1]):
                values[k] -= (x[col_indices[k]] * db1) // 127
                
    return values, b1, W2, b2


# --------------------------
# Core Inference Function
# --------------------------
@njit(parallel=True, fastmath=True, cache=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD):
    n = X_i8.shape[0]
    num_classes = len(b2)
    pred = np.empty(n, np.int64)
    ee = 0
    for i in prange(n):
        x = X_i8[i]
        h = np.zeros(H, np.int64)
        
        # Forward Pass - Layer 1
        for j in range(H):
            acc = b1[j]
            for k in range(indptr[j], indptr[j+1]):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)
            
        # Forward Pass - Layer 2 (Logits)
        logits = b2.copy()
        for j in range(H):
            if h[j]:
                for c in range(num_classes):
                    logits[c] += h[j] * W2[j, c]
        
        # Softmax / Confidence Approximation (ใช้ค่า d แทน e^d)
        min_l = logits.min()
        sum_d = 0
        max_l = logits[0]
        for c in range(num_classes):
            d = min(127, max(0, (logits[c] - min_l) >> 1))
            sum_d += d
            if logits[c] > max_l:
                max_l = logits[c]
                
        conf = (max_l * 100) // max(sum_d, 1)
        pred[i] = np.argmax(logits)
        
        if conf >= CONF_THRESHOLD:
            ee += 1
            
    return pred, ee / n


# --------------------------
# Optional Data Stream Generator
# --------------------------
def data_stream(n=1000):
    # ใช้ N_FEATURES และ N_CLASSES ที่ถูกกำหนดไว้ใน Global Config
    X = np.random.randn(n, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, size=n).astype(np.int32)
    yield X, y
