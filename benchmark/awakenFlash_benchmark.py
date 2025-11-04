# awakenFlash_benchmark_faiss_ready.py
import subprocess
import sys

# ===============================
# ติดตั้ง FAISS ถ้าไม่เจอ
# ===============================
try:
    import faiss
except ModuleNotFoundError:
    print("📦 FAISS not found. Installing faiss-cpu...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
    import faiss

# ===============================
# โค้ด Non-Logic Fast Enhanced Benchmark
# ===============================
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.metrics import accuracy_score, f1_score

# Config
dim = 128
batch_size = 32
k_neighbors = 5
max_workers = 8

# Dummy dataset
np.random.seed(42)
num_samples = 10000
train_embeddings = np.random.rand(num_samples, dim).astype('float32')
train_labels = np.random.randint(0, 2, size=num_samples)
test_embeddings = np.random.rand(num_samples, dim).astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(dim)
index.add(train_embeddings)

# Mini-batch function
def predict_batch(batch_emb):
    D, I = index.search(batch_emb, k_neighbors)
    preds = np.array([np.bincount(I_row).argmax() for I_row in I])
    return preds

def chunk_embeddings(embeddings, size):
    for i in range(0, len(embeddings), size):
        yield embeddings[i:i+size]

# Benchmark
start_time = time.time()
batches = list(chunk_embeddings(test_embeddings, batch_size))
predictions = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(predict_batch, batches))
    for r in results:
        predictions.extend(r)

predictions = np.array(predictions)
true_labels = np.random.randint(0, 2, size=len(predictions))

acc = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
end_time = time.time()

print(f"✅ Non-Logic Fast Enhanced Acc={acc:.4f} | F1={f1:.4f} | Time={end_time-start_time:.2f}s")
