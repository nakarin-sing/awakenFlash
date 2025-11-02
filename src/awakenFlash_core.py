#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash core — train + infer functions (INT8)
"""

import numpy as np
from numba import njit, prange

# --------------------------
# Global Config
# --------------------------
N_SAMPLES = 10_000
N_FEATURES = 20
N_CLASSES = 2
B = 1024
H = 64
CONF_THRESHOLD = 80
LS = 0.1

# --------------------------
# Core Train Function
# --------------------------
@njit(parallel=True, fastmath=True, cache=True)
def train_step(X_i8, y, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS):
    n = X_i8.shape[0]
    for i in prange(n):  # ใช้ step=1 เพื่อให้ Numba รองรับ
        x = X_i8[i]
        target = y[i]
        h = np.zeros(H, np.int64)
        for j in range(H):
            acc = b1[j]
            for k in range(indptr[j], indptr[j+1]):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)
        logits = b2.copy()
        for j in range(H):
            if h[j]:
                for c in range(len(b2)):
                    logits[c] += h[j] * W2[j, c]
        min_l = logits.min()
        sum_e = 0
        max_l = logits[0]
        for c in range(len(logits)):
            d = min(127, max(0, (logits[c] - min_l) >> 1))
            sum_e += d
            if logits[c] > max_l:
                max_l = logits[c]
        conf = (max_l * 100) // max(sum_e, 1)
        pred = np.argmax(logits)
        if conf < CONF_THRESHOLD:
            pred = target
        tgt = np.zeros(len(b2), np.int64)
        tgt[pred] = 127
        tgt = ((1 - LS) * tgt + LS * 127 // len(b2)).astype(np.int64)
        prob = np.zeros(len(b2), np.int64)
        for c in range(len(b2)):
            prob[c] = min(127, max(0, (logits[c] - min_l) >> 1))
        dL = np.clip((prob - tgt) // 127, -5, 5)
        for c in range(len(b2)):
            db = dL[c]
            b2[c] -= db
            for j in range(H):
                if h[j]:
                    W2[j, c] -= (h[j] * db) // 127
        for j in range(H):
            dh = 0
            for c in range(len(b2)):
                if h[j]:
                    dh += dL[c] * W2[j, c]
            db1 = dh // len(b2)
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
    pred = np.empty(n, np.int64)
    ee = 0
    for i in prange(n):
        x = X_i8[i]
        h = np.zeros(H, np.int64)
        for j in range(H):
            acc = b1[j]
            for k in range(indptr[j], indptr[j+1]):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)
        logits = b2.copy()
        for j in range(H):
            if h[j]:
                for c in range(len(b2)):
                    logits[c] += h[j] * W2[j, c]
        min_l = logits.min()
        sum_e = 0
        max_l = logits[0]
        for c in range(len(logits)):
            d = min(127, max(0, (logits[c] - min_l) >> 1))
            sum_e += d
            if logits[c] > max_l:
                max_l = logits[c]
        conf = (max_l * 100) // max(sum_e, 1)
        pred[i] = np.argmax(logits)
        if conf >= CONF_THRESHOLD:
            ee += 1
    return pred, ee / n


# --------------------------
# Optional Data Stream Generator
# --------------------------
def data_stream(n=1000):
    X = np.random.randn(n, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, size=n).astype(np.int32)
    yield X, y
