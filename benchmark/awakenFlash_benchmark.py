import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ========================
# DATA PREP (dummy example)
# ========================
X_train = np.random.rand(50000, 54).astype(np.float32)
y_train = np.random.randint(0, 7, size=(50000,))
X_test  = np.random.rand(10000, 54).astype(np.float32)
y_test  = np.random.randint(0, 7, size=(10000,))

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# ========================
# NON-LOGIC v3: Mini-Batch Approximate NN
# ========================
BATCH_SIZE = 2000
TOP_K = 10
SAMPLE_SUBSET = 5000  # ใช้ subset ของ training สำหรับ approx

preds = []

for start in range(0, X_test.shape[0], BATCH_SIZE):
    end = start + BATCH_SIZE
    batch = X_test[start:end]

    # --- Random subset of training data for fast approximate NN ---
    idx_subset = np.random.choice(X_train.shape[0], SAMPLE_SUBSET, replace=False)
    X_sub = X_train[idx_subset]
    y_sub = y_train[idx_subset]

    # --- Euclidean distance ---
    dists = np.linalg.norm(X_sub[None,:,:] - batch[:,None,:], axis=2)  # shape (batch, subset)
    neighbors = np.argsort(dists, axis=1)[:,:TOP_K]

    # --- Weighted voting ---
    for nbrs in neighbors:
        lbls = y_sub[nbrs]
        weights = np.linspace(0.5,1.0,TOP_K)
        scores = np.zeros(np.max(y_train)+1)
        for w,lbl in zip(weights, lbls):
            scores[lbl] += w
        preds.append(np.argmax(scores))

preds = np.array(preds)

acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds, average='macro')

print("Non-Logic v3 (Fast) Acc =", acc)
print("Non-Logic v3 (Fast) F1  =", f1)
