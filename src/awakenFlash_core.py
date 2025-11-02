#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash core — optimized train + infer
"""

import numpy as np
from numba import njit, prange

# ===================== INFERENCE =====================
@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD):
    n = X_i8.shape[0]
    pred = np.empty(n, np.int64)
    ee_count = 0

    H = b1.shape[0]
    C = b2.shape[0]

    for i in prange(n):
        x = X_i8[i]
        h = np.zeros(H, np.int32)
        for j in range(H):
            acc = b1[j]
            for k in range(indptr[j], indptr[j+1]):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)

        logits = b2.copy()
        for j in np.nonzero(h)[0]:
            for c in range(C):
                logits[c] += h[j] * W2[j, c]

        min_l = np.min(logits)
        sum_e = 0
        max_l = logits[0]
        for c in range(C):
            d = min(127, max(0, (logits[c]-min_l)>>1))
            e = lut_exp[d]
            sum_e += e
            if logits[c] > max_l:
                max_l = logits[c]
        conf = max_l*100 // max(sum_e,1)
        pred[i] = np.argmax(logits)
        if conf >= CONF_THRESHOLD:
            ee_count += 1
    return pred, ee_count / n

# ===================== TRAIN =====================
@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def train_step(X_i8, y, values, col_indices, indptr, b1, W2, b2, B=1024, CONF_THRESHOLD=80, LS=0.006, lut_exp=None):
    n = X_i8.shape[0]
    H = b1.shape[0]
    C = b2.shape[0]
    n_batch = (n + B - 1) // B

    for i in prange(n_batch):
        start = i * B
        end = min(start + B, n)
        xb, yb = X_i8[start:end], y[start:end]
        ns = xb.shape[0]

        for s in range(ns):
            x = xb[s]
            h = np.zeros(H, np.int32)
            for j in range(H):
                acc = b1[j]
                p, q = indptr[j], indptr[j+1]
                for k in range(p,q):
                    acc += x[col_indices[k]] * values[k]
                h[j] = max(acc >> 5, 0)

            logits = b2.copy()
            for j in np.nonzero(h)[0]:
                for c in range(C):
                    logits[c] += h[j]*W2[j,c]

            min_l = np.min(logits)
            max_l = logits[0]
            sum_e = 0
            for c in range(C):
                d = min(127, max(0, (logits[c]-min_l)>>1))
                e = lut_exp[d] if lut_exp is not None else d
                sum_e += e
                if logits[c] > max_l:
                    max_l = logits[c]

            conf = max_l*100 // max(sum_e,1)
            pred = np.argmax(logits)
            target = yb[s] if conf < CONF_THRESHOLD else pred

            # target vector
            tgt = np.zeros(C, np.int32)
            tgt[target] = 127
            tgt = ((1-LS)*tgt + LS*127//3).astype(np.int32)

            # prob vector
            prob = np.zeros(C, np.int32)
            for c in range(C):
                d = min(127,max(0,(logits[c]-min_l)>>1))
                prob[c] = lut_exp[d] if lut_exp is not None else d
            sum_prob = np.sum(prob)
            if sum_prob>0:
                prob = prob*127 // sum_prob

            dL = np.clip((prob - tgt)//127, -5, 5)

            # update b2 & W2
            for c in range(C):
                db = dL[c] // ns
                b2[c] -= db
                for j in np.nonzero(h)[0]:
                    W2[j,c] -= (h[j]*db)//127

            # update b1 & values
            for j in range(H):
                dh = 0
                for c in range(C):
                    dh += dL[c]*W2[j,c] if h[j]>0 else 0
                db1 = dh//ns
                b1[j] -= db1
                p,q = indptr[j], indptr[j+1]
                for k in range(p,q):
                    values[k] -= (x[col_indices[k]]*db1)//127
    return values, b1, W2, b2
