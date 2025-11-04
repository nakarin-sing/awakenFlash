# awakenFlash_benchmark_fast.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
import time

# ===========================
# Dataset simulation
# ===========================
print("📦 Loading dataset...")
X, y = make_classification(n_samples=100000, n_features=54, n_informative=20,
                           n_classes=7, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into chunks
chunk_size = 10000
chunks = [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, X.shape[0], chunk_size)]

# ===========================
# Non-Logic "Fast" Benchmark
# ===========================
print("🚀 Running Non-Logic Fast Benchmark...")

results = []

for i, (X_chunk, y_chunk) in enumerate(chunks, 1):
    start = time.time()
    
    # ใช้ NearestNeighbors แทน FAISS
    nn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X_chunk)
    distances, indices = nn.kneighbors(X_chunk)
    
    # Fake predictions แบบเร็วๆ (ใช้ majority vote ของ neighbors)
    y_pred = []
    for idx_list in indices:
        neighbors = y_chunk[idx_list]
        counts = np.bincount(neighbors)
        y_pred.append(np.argmax(counts))
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_chunk, y_pred)
    f1 = f1_score(y_chunk, y_pred, average='macro')
    
    elapsed = time.time() - start
    print(f"Chunk {i} | Non-Logic Fast Acc={acc:.4f} | F1={f1:.4f} | Time={elapsed:.2f}s")
    results.append((acc, f1, elapsed))

print("✅ Benchmark completed in {:.2f}s".format(sum(r[2] for r in results)))
