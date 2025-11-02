#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash core — train + infer functions (INT8) (improved)
"""

import numpy as np
from numba import njit, prange

# --------------------------
# Defaults (can be overridden by caller)
# --------------------------
# NOTE: benchmark passes these explicitly to keep things explicit for numba.
# N_SAMPLES, N_FEATURES, N_CLASSES are only defaults for local testing
N_SAMPLES = 10000
N_FEATURES = 32
N_CLASSES = 3
# safe defaults if not provided
B_DEFAULT = 1024
LS_DEFAULT = 0.006
CONF_DEFAULT = 80

# --------------------------
# LUT-softmax helper
# --------------------------
@njit(cache=True)
def _lut_softmax_probs_int64(logits, lut):
    # logits: 1d int64
    L = logits.shape[0]
    mn = logits[0]
    for i in range(1, L):
        if logits[i] < mn:
            mn = logits[i]
    s = 0
    probs = np.empty(L, np.int32)
    for i in range(L):
        d = (logits[i] - mn) >> 1
        if d < 0:
            e = 1
        elif d >= lut.shape[0]:
            e = int(lut[lut.shape[0] - 1])
        else:
            e = int(lut[d])
        probs[i] = e
        s += e
    if s == 0:
        s = 1
    for i in range(L):
        probs[i] = (probs[i] * 127) // s
    return probs

# --------------------------
# TRAIN STEP (numba optimized)
# signature: many args so numba infers types correctly
# --------------------------
@njit(parallel=True, cache=True, nogil=True, fastmath=True)
def train_step(X_i8, y, values, col_indices, indptr, b1, W2, b2,
               B_local, H_local, CONF_threshold, LS_local, lut_local):
    n = X_i8.shape[0]
    # parallelize by batches (n_batches = ceil(n/B_local))
    n_batches = (n + B_local - 1) // B_local
    for bi in prange(n_batches):
        start = bi * B_local
        end = start + B_local
        if end > n:
            end = n
        # per-sample updates inside batch (keeps numba happy)
        for i in range(start, end):
            x = X_i8[i]
            y_t = y[i]
            # forward hidden
            h = np.zeros(H_local, np.int64)
            for j in range(H_local):
                acc = b1[j]
                p = indptr[j]
                q = indptr[j+1]
                for k in range(p, q):
                    acc += np.int64(x[col_indices[k]]) * np.int64(values[k])
                if acc > 0:
                    h[j] = acc >> 5
                else:
                    h[j] = 0
            # logits
            Lc = b2.shape[0]
            logits = np.zeros(Lc, np.int64)
            for j in range(H_local):
                if h[j] > 0:
                    for c in range(Lc):
                        logits[c] += h[j] * np.int64(W2[j, c])
            # probs via LUT
            probs = _lut_softmax_probs_int64(logits, lut_local)
            # compute confidence (use same LUT denom logic)
            mn = logits[0]
            for c in range(1, Lc):
                if logits[c] < mn:
                    mn = logits[c]
            sum_e = 0
            max_l = logits[0]
            for c in range(Lc):
                d = (logits[c] - mn) >> 1
                if d < 0:
                    e = 1
                elif d >= lut_local.shape[0]:
                    e = int(lut_local[lut_local.shape[0]-1])
                else:
                    e = int(lut_local[d])
                sum_e += e
                if logits[c] > max_l:
                    max_l = logits[c]
            conf = (max_l * 100) // max(1, sum_e)
            # pseudo-label / teacher-choice
            chosen = y_t if conf < CONF_threshold else 0
            # if confident use predicted argmax
            if conf >= CONF_threshold:
                # argmax
                best = 0
                bv = logits[0]
                for c in range(1, Lc):
                    if logits[c] > bv:
                        bv = logits[c]; best = c
                chosen = best
            # target smoothing
            tgt = np.zeros(Lc, np.int64)
            tgt[chosen] = 127
            for c in range(Lc):
                tgt[c] = int((1 - LS_local) * tgt[c] + LS_local * (127 // Lc))
            # gradient-ish delta (integer)
            dL = np.zeros(Lc, np.int32)
            for c in range(Lc):
                dL[c] = (probs[c] - tgt[c]) // 127
                if dL[c] > 6: dL[c] = 6
                if dL[c] < -6: dL[c] = -6
            # update W2, b2
            for c in range(Lc):
                db = dL[c]
                b2[c] -= db
                for j in range(H_local):
                    if h[j] > 0:
                        W2[j, c] -= (h[j] * db) // 127
            # update W1 values and b1
            for j in range(H_local):
                dh = 0
                if h[j] > 0:
                    for c in range(Lc):
                        dh += dL[c] * np.int64(W2[j, c])
                db1 = dh // max(1, Lc)
                b1[j] -= db1
                p = indptr[j]; q = indptr[j+1]
                for k in range(p, q):
                    values[k] -= (np.int64(x[col_indices[k]]) * db1) // 127
    return values, b1, W2, b2

# --------------------------
# INFER (numba optimized)
# --------------------------
@njit(cache=True, nogil=True, fastmath=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, lut_local, CONF_threshold):
    n = X_i8.shape[0]
    pred = np.empty(n, np.int64)
    ee = 0
    H_local = b1.shape[0]
    Lc = b2.shape[0]
    for i in range(n):
        x = X_i8[i]
        h = np.zeros(H_local, np.int64)
        for j in range(H_local):
            acc = b1[j]
            p = indptr[j]; q = indptr[j+1]
            for k in range(p, q):
                acc += np.int64(x[col_indices[k]]) * np.int64(values[k])
            if acc > 0:
                h[j] = acc >> 5
            else:
                h[j] = 0
        logits = np.zeros(Lc, np.int64)
        for j in range(H_local):
            if h[j] > 0:
                for c in range(Lc):
                    logits[c] += h[j] * np.int64(W2[j, c])
        # conf
        mn = logits[0]
        for c in range(1, Lc):
            if logits[c] < mn:
                mn = logits[c]
        sum_e = 0
        max_l = logits[0]
        for c in range(Lc):
            d = (logits[c] - mn) >> 1
            if d < 0:
                e = 1
            elif d >= lut_local.shape[0]:
                e = int(lut_local[lut_local.shape[0]-1])
            else:
                e = int(lut_local[d])
            sum_e += e
            if logits[c] > max_l:
                max_l = logits[c]
        conf = (max_l * 100) // max(1, sum_e)
        # argmax
        best = 0; bv = logits[0]
        for c in range(1, Lc):
            if logits[c] > bv:
                bv = logits[c]; best = c
        pred[i] = best
        if conf >= CONF_threshold:
            ee += 1
    return pred, ee / n

# --------------------------
# Optional pure-python data_stream for debugging (not numba)
# --------------------------
def data_stream(n=1000, seed=42, features=32, classes=3, chunk=100000):
    rng = np.random.default_rng(seed)
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        X = rng.standard_normal((size, features)).astype(np.float32)
        if features >= 16:
            X = np.hstack([X, (X[:, :8] * X[:, 8:16])])
        y = rng.integers(0, classes, size, dtype=np.int64)
        yield X, y
