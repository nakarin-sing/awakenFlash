#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
awakenFlash core — train + infer functions
"""

import numpy as np
from numba import njit, prange

# ===================== TRAIN + INFER =====================
@njit(parallel=True, fastmath=True, cache=True)
def train_step(X_i8, y, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD, LS):
    n = X_i8.shape[0]
    B = 1024  # batch-size internal

    for start in range(0, n, B):  # ใช้ range ธรรมดา
        end = min(start + B, n)
        xb = X_i8[start:end]
        yb = y[start:end]
        ns = xb.shape[0]

        for s in range(ns):
            x = xb[s]
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
                        logits[c] += h[j] * W2[j,c]

            min_l = logits.min()
            sum_e = 0
            max_l = logits[0]
            for c in range(len(logits)):
                d = min(127,max(0,(logits[c]-min_l)>>1))
                e = min(255,d+1)
                sum_e += e
                if logits[c] > max_l:
                    max_l = logits[c]

            conf = (max_l*100)//max(sum_e,1)
            pred = np.argmax(logits)
            target = yb[s] if conf < CONF_THRESHOLD else pred
            tgt = np.zeros(len(b2), np.int64)
            tgt[target] = 127
            tgt = ((1-LS)*tgt + LS*127//3).astype(np.int64)

            prob = np.zeros(len(b2), np.int64)
            for c in range(len(b2)):
                prob[c] = min(127,max(0,(logits[c]-min_l)>>1))

            dL = np.clip((prob-tgt)//127,-5,5)
            for c in range(len(b2)):
                db = dL[c]//ns
                b2[c] -= db
                for j in range(H):
                    if h[j]:
                        W2[j,c] -= (h[j]*db)//127

            for j in range(H):
                dh = 0
                for c in range(len(b2)):
                    if h[j]:
                        dh += dL[c]*W2[j,c]
                db1 = dh//ns
                b1[j] -= db1
                for k in range(indptr[j],indptr[j+1]):
                    values[k] -= (x[col_indices[k]]*db1)//127

    return values,b1,W2,b2


@njit(parallel=True, fastmath=True, cache=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, H, CONF_THRESHOLD):
    n = X_i8.shape[0]
    pred = np.empty(n,np.int64)
    ee = 0
    for i in prange(n):
        x = X_i8[i]
        h = np.zeros(H,np.int64)
        for j in range(H):
            acc = b1[j]
            for k in range(indptr[j],indptr[j+1]):
                acc += x[col_indices[k]]*values[k]
            h[j]=max(acc>>5,0)
        logits = b2.copy()
        for j in range(H):
            if h[j]:
                for c in range(len(b2)):
                    logits[c]+=h[j]*W2[j,c]
        min_l = logits.min()
        sum_e=0
        max_l=logits[0]
        for c in range(len(logits)):
            d = min(127,max(0,(logits[c]-min_l)>>1))
            sum_e+=d
            if logits[c]>max_l: max_l=logits[c]
        conf=(max_l*100)//max(sum_e,1)
        pred[i]=np.argmax(logits)
        if conf>=CONF_THRESHOLD:
            ee+=1
    return pred, ee/n
