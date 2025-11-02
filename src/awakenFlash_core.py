import numpy as np
from numba import njit, prange

# --- LUT สำหรับ softmax-like ---
lut_exp = np.ascontiguousarray(np.array([
    1,1,1,1,1,2,2,2,2,3,3,3,4,4,5,6,7,8,9,10,
    11,13,15,17,19,22,25,28,32,36,41,46,52,59,67,76,
    86,97,110,124,140,158,179,202,228,255
] + [255]*88, np.uint8)[:128])

# --- Inference function ---
@njit(parallel=True, cache=True)
def infer(X_i8, values, col_indices, indptr, b1, W2, b2, lut_exp, CONF_THRESHOLD):
    n_samples = X_i8.shape[0]
    preds = np.empty(n_samples, np.int64)
    early_exit = 0
    H = b1.shape[0]

    for s in prange(n_samples):
        x = X_i8[s]
        h = np.zeros(H, np.int64)
        for j in range(H):
            acc = b1[j]
            for k in range(indptr[j], indptr[j+1]):
                acc += x[col_indices[k]] * values[k]
            h[j] = max(acc >> 5, 0)
        logits = b2.copy()
        for j in range(H):
            if h[j]:
                for c in range(3):
                    logits[c] += h[j] * W2[j, c]

        min_l = logits.min()
        sum_e = 0
        max_l = logits[0]
        for c in range(3):
            d = min(127, max(0, (logits[c]-min_l)>>1))
            e = lut_exp[d]
            sum_e += e
            if logits[c] > max_l: max_l = logits[c]

        conf = (max_l*100)//max(sum_e,1)
        pred = np.argmax(logits)
        if conf >= CONF_THRESHOLD:
            early_exit += 1
        preds[s] = pred

    ee_ratio = early_exit / n_samples
    return preds, ee_ratio
