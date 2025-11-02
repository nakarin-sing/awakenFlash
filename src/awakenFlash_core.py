# src/awakenFlash_core.py
import numpy as np
from numba import njit, prange

@njit(cache=True)
def lut_softmax(logits, lut):
    min_l = logits.min()
    sum_e = 0
    max_l = logits[0]
    for i in range(logits.shape[0]):
        d = min(127, max(0, (logits[i] - min_l) >> 1))
        if logits[i] > max_l: max_l = logits[i]
        sum_e += lut[d]
    return (max_l * 100) // max(sum_e, 1)

@njit(parallel=True, cache=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, lut, conf_thr):
    n, H = X_i8.shape[0], b1.shape[0]
    pred = np.empty(n, np.int64)
    ee = 0
    for i in prange(n):
        x = X_i8[i]
        h = np.zeros(H, np.int64)
        for j in range(H):
            acc = b1[j]
            p, q = indptr[j], indptr[j+1]
            for k in range(p, q):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)
        logits = b2.copy()
        for j in range(H):
            if h[j]:
                for c in range(W2.shape[1]):
                    logits[c] += h[j] * W2[j, c]
        conf = lut_softmax(logits, lut)
        pred[i] = np.argmax(logits)
        if conf >= conf_thr: ee += 1
    return pred, ee / n
