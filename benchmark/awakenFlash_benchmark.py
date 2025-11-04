import subprocess
import sys
import time

# =======================
# ติดตั้ง faiss ถ้าไม่มี
# =======================
try:
    import faiss
except ModuleNotFoundError:
    print("📦 FAISS not found. Installing faiss-cpu...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# =======================
# Mock dataset (แทนด้วย dataset จริง)
# =======================
num_samples = 100000
num_features = 128
num_classes = 5
chunks = 10

np.random.seed(42)
X = np.random.rand(num_samples, num_features).astype('float32')
y_true = np.random.randint(0, num_classes, size=num_samples)

# =======================
# เริ่มต้น centroids แบบสุ่ม
# =======================
centroids = np.random.rand(num_classes, num_features).astype('float32')

# =======================
# Non-Logic Fast++ Function
# =======================
def non_logic_dynamic_predict_update(X_chunk, centroids, alpha=0.3):
    """
    alpha: learning rate ของ centroids update
    """
    d = X_chunk.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    _, labels = index.search(X_chunk, 1)
    
    # Update centroids แบบ dynamic (moving average)
    for c in range(centroids.shape[0]):
        mask = labels.flatten() == c
        if np.any(mask):
            centroids[c] = (1-alpha)*centroids[c] + alpha*X_chunk[mask].mean(axis=0)
    return labels.flatten(), centroids

# =======================
# Benchmark 10 Chunks
# =======================
chunk_size = num_samples // chunks
start_total = time.time()

for i in range(chunks):
    start_chunk = time.time()
    X_chunk = X[i*chunk_size:(i+1)*chunk_size]
    y_chunk = y_true[i*chunk_size:(i+1)*chunk_size]
    
    y_pred, centroids = non_logic_dynamic_predict_update(X_chunk, centroids)
    
    acc = accuracy_score(y_chunk, y_pred)
    f1 = f1_score(y_chunk, y_pred, average='weighted')
    end_chunk = time.time()
    print(f"Chunk {i+1} | Non-Logic Fast++ Acc={acc:.4f} | F1={f1:.4f} | Time={end_chunk-start_chunk:.2f}s")

end_total = time.time()
print(f"✅ Benchmark completed in {end_total-start_total:.2f}s")
