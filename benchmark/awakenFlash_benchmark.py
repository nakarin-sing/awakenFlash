import numpy as np
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
import time

# ====== Config ======
N_SAMPLES = 100000
N_FEATURES = 50
N_CLASSES = 7
CHUNK_SIZE = 10000
DIM_REDUCTION = 32
N_LIST = 100
N_PROBE = 5
K_NEIGHBORS = 5
ALPHA_ONLINE = 0.3
BETA_SELF_CORRECT = 0.2
N_UNIVERSES = 5
np.random.seed(42)

# ====== Dataset ======
X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES, random_state=42)
X = StandardScaler().fit_transform(X).astype('float32')

# ====== PCA Reduction ======
def reduce_dim(X, dim=DIM_REDUCTION):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :dim] * S[:dim]

X_reduced = reduce_dim(X, DIM_REDUCTION)

# ====== FAISS Index for Multiverse ======
indexes = []
for u in range(N_UNIVERSES):
    idx = faiss.index_factory(DIM_REDUCTION, f"IVF{N_LIST},Flat")
    idx.train(X_reduced)
    idx.add(X_reduced)
    idx.nprobe = N_PROBE
    indexes.append(idx)

# ====== Online Update with Temporal + Self-Correction ======
def update_labels(indexes, X_chunk, y_chunk, y_memory, alpha=ALPHA_ONLINE, beta=BETA_SELF_CORRECT):
    # simulate multiverse reasoning
    y_updated = np.zeros_like(y_chunk)
    for i, x in enumerate(X_chunk):
        votes = np.zeros(N_CLASSES)
        for u, idx in enumerate(indexes):
            D, I = idx.search(x.reshape(1, -1), K_NEIGHBORS)
            neighbor_labels = y_memory[I[0]]
            counts = np.bincount(neighbor_labels, minlength=N_CLASSES)
            votes += counts / (u+1)  # temporal weighting
        majority_vote = np.argmax(votes)
        if np.random.rand() < alpha:
            y_updated[i] = majority_vote
        else:
            if np.random.rand() < beta:
                y_updated[i] = majority_vote
            else:
                y_updated[i] = y_chunk[i]
    return y_updated

# ====== Prediction ======
def faiss_predict(indexes, X_query, y_memory):
    y_pred = np.zeros(X_query.shape[0], dtype=int)
    for i, x in enumerate(X_query):
        votes = np.zeros(N_CLASSES)
        for u, idx in enumerate(indexes):
            D, I = idx.search(x.reshape(1, -1), K_NEIGHBORS)
            votes += np.bincount(y_memory[I[0]], minlength=N_CLASSES) / (u+1)
        y_pred[i] = np.argmax(votes)
    return y_pred

# ====== Benchmark ======
start = time.time()
y_memory = y.copy()
acc_list, f1_list = [], []

n_chunks = N_SAMPLES // CHUNK_SIZE
for i in range(n_chunks):
    X_chunk = X_reduced[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
    y_chunk = y[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]

    # predict
    y_pred = faiss_predict(indexes, X_chunk, y_memory)

    # evaluate
    acc = accuracy_score(y_chunk, y_pred)
    f1 = f1_score(y_chunk, y_pred, average='macro')
    acc_list.append(acc)
    f1_list.append(f1)
    print(f"Chunk {i+1} | Acc={acc:.3f}, F1={f1:.3f}")

    # online update
    y_memory[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] = update_labels(indexes, X_chunk, y_chunk, y_memory)

print("Average | Acc={:.3f}, F1={:.3f}".format(np.mean(acc_list), np.mean(f1_list)))
print("Elapsed Time: {:.2f}s".format(time.time()-start))
