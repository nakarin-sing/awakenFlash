# awakenFlash_benchmark.py — Non-Logic Fast+ Learning (GitHub Actions Ready)
import os
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ─── Setup ───
np.random.seed(42)  # reproducible
NUM_CHUNKS = 10
NUM_SAMPLES = 100
NUM_FEATURES = 20
NUM_CLASSES = 3

# สร้าง dataset ตัวอย่าง
X_chunks = [np.random.rand(NUM_SAMPLES, NUM_FEATURES) for _ in range(NUM_CHUNKS)]
y_chunks = [np.random.randint(0, NUM_CLASSES, NUM_SAMPLES) for _ in range(NUM_CHUNKS)]

predictions_total = []
true_total = []

start_time = time.time()
for idx, (X, y_true) in enumerate(zip(X_chunks, y_chunks), 1):
    chunk_start = time.time()
    
    # ─── Non-Logic Fast+ Learning ───
    means = X.mean(axis=0)
    stds = X.std(axis=0) + 1e-6
    X_norm = (X - means) / stds
    
    # Pseudo-learning: prediction by argmax mod number of classes
    pred = X_norm.argmax(axis=1) % NUM_CLASSES
    
    predictions_total.extend(pred)
    true_total.extend(y_true)
    
    acc_chunk = accuracy_score(y_true, pred)
    f1_chunk = f1_score(y_true, pred, average='weighted')
    
    print(f"Chunk {idx} | Non-Logic Fast+ Learning Acc={acc_chunk:.4f} | F1={f1_chunk:.4f} | Time≈{time.time()-chunk_start:.2f}s")

# ─── Final metrics ───
acc_total = accuracy_score(true_total, predictions_total)
f1_total = f1_score(true_total, predictions_total, average='weighted')
print(f"\n✅ Benchmark completed in {time.time()-start_time:.2f}s")
print(f"Total | Acc={acc_total:.4f} | F1={f1_total:.4f}")
