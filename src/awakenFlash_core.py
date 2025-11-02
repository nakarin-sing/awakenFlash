#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash core — train + infer functions (INT8) [v2.0: Max Speed Inference]
"""

import numpy as np
from numba import njit, prange

# --------------------------
# Global Config (Unchanged from v1.8)
# --------------------------
N_SAMPLES = 100_000 
N_FEATURES = 40     
N_CLASSES = 3       
H = 448            
CONF_THRESHOLD = 80
LS = 0.006          

# --------------------------
# Core Train Function (unchanged logic from v1.8, now with LUT_EXP usage)
# --------------------------
# LUT_EXP ที่ถูกกำหนดไว้ใน Benchmark (แต่ควรอยู่ใน Core) ถูกจำลองไว้ใน Train_step 
# เพื่อให้ Logic ถูกต้อง

# จำลอง LUT_EXP ที่ถูกส่งเข้ามา (ใน benchmark มีการกำหนดไว้)
# NOTE: โค้ดที่รันจริงจะต้องมี lut_exp ถูกส่งเข้ามาใน train_step ด้วย
# แต่เพื่อให้โค้ดนี้รันได้ใน Numba Standalone จะต้องมีการกำหนดค่าตายตัว
lut_exp = np.ascontiguousarray(np.array([ 1,1,1,1,1,2,2,2,2,3,3,3,4,4,5,6,7,8,9,10, 
 11,13,15,17,19,22,25,28,32,36,41,46,52,59,67,76, 
 86,97,110,124,140,158,179,202,228,255 ] + [255]*88, np.uint8)[:128], dtype=np.int64)

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
def train_step(X_i8, y, values, col_indices, indptr, b1, W2, b2, B, H, CONF_THRESHOLD, LS, lut_exp):
    n = X_i8.shape[0]
    num_classes = 3 # N_CLASSES
    n_batch = (n + B - 1) // B
    
    for i in prange(n_batch):
        start = i * B
        end = min(start + B, n)
        if start >= n: break
            
        # Batching Logic in Train Step
        xb, yb = X_i8[start:end], y[start:end]
        ns = xb.shape[0]
        
        for s in range(ns):
            x = xb[s]; h = np.zeros(H, np.int64)
            
            # Layer 1
            for j in range(H):
                acc = b1[j]; p, q = indptr[j], indptr[j+1]
                for k in range(p, q):
                    acc += x[col_indices[k]] * values[k]
                h[j] = max(acc >> 5, 0)
                
            # Layer 2
            logits = b2.copy()
            for j in range(H):
                if h[j]:
                    for c in range(num_classes):
                        logits[c] += h[j] * W2[j, c]
                        
            # Confidence Check (ใช้ LUT_EXP)
            min_l = logits.min(); sum_e = 0; max_l = logits[0]
            for c in range(num_classes):
                d = min(127, max(0, (logits[c] - min_l) >> 1))
                e = lut_exp[d]; sum_e += e
                if logits[c] > max_l: max_l = logits[c]
            
            conf = (max_l * 100) // max(sum_e, 1)
            pred = np.argmax(logits)
            
            # Label Smoothing & Early Exit Target
            target = yb[s] if conf < CONF_THRESHOLD else pred
            tgt = np.zeros(num_classes, np.int64); tgt[target] = 127
            tgt = ((1 - LS) * tgt + LS * 127 // num_classes).astype(np.int64)
            
            # Probability Vector
            prob = np.zeros(num_classes, np.int64); sum_e = 0
            for c in range(num_classes):
                d = min(127, max(0, (logits[c] - min_l) >> 1))
                prob[c] = lut_exp[d]; sum_e += prob[c]
            prob = (prob * 127) // max(sum_e, 1)
            
            # Backpropagation (dL, Update W2, b2, W1, b1)
            dL = np.clip((prob - tgt) // 127, -5, 5) 
            for c in range(num_classes):
                db = dL[c] // ns; b2[c] -= db
                for j in range(H):
                    if h[j]: W2[j, c] -= (h[j] * db) // 127
            
            for j in range(H):
                dh = 0
                for c in range(num_classes):
                    if h[j]: dh += dL[c] * W2[j, c]
                db1 = dh // ns; b1[j] -= db1
                p, q = indptr[j], indptr[j+1]
                for k in range(p, q):
                    values[k] -= (x[col_indices[k]] * db1) // 127
                    
    return values, b1, W2, b2

# --------------------------
# Core Inference Function (MAX SPEED - NO CONFIDENCE CHECK)
# --------------------------
# **NOTE:** ฟังก์ชันนี้จะต้องรับ lut_exp และ CONF_THRESHOLD เข้ามาตามที่ Benchmark เรียกใช้
@njit(parallel=True, fastmath=True, cache=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD): 
    n = X_i8.shape[0]
    num_classes = len(b2)
    pred = np.empty(n, np.int64)
    ee = 0 # Early Exit ถูกปิดการใช้งานเพื่อความเร็วสูงสุด
    H_val = len(b1) # Use H from b1 size
    
    for i in prange(n):
        x = X_i8[i]
        h = np.zeros(H_val, np.int64)
        
        # Forward Pass - Layer 1 (W1 * X + b1)
        for j in range(H_val):
            acc = b1[j]
            for k in range(indptr[j], indptr[j+1]):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)
            
        # Forward Pass - Layer 2 (W2 * H + b2) -> Logits
        logits = b2.copy()
        for j in range(H_val):
            if h[j]:
                for c in range(num_classes):
                    logits[c] += h[j] * W2[j, c]
        
        # Prediction Only 
        pred[i] = np.argmax(logits)
            
    return pred, ee / n # ee/n จะเท่ากับ 0.0 เสมอ
