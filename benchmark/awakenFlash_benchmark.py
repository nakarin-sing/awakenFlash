# awakenFlash_benchmark_fixed.py

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# ===============================
# Non-Logic Boost Function
# ===============================
def non_logic_boost(X):
    # เพิ่มฟีเจอร์ใหม่แบบง่าย ๆ
    return np.hstack([
        X,            # Original
        X**2,         # Quadratic
        np.sin(X),    # Sine
        np.prod(X[:, :2], axis=1, keepdims=True)  # Interaction first 2 cols
    ])

# ===============================
# Online MLP for memory-based learning
# ===============================
class OnlineMLP:
    def __init__(self, input_dim, hidden_dim=32, output_dim=3, lr=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        exp_z = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def update(self, X, y, epochs=1):
        y_onehot = np.eye(self.b2.shape[1])[y]
        for _ in range(epochs):
            probs = self.forward(X)
            error = probs - y_onehot
            dW2 = self.a1.T @ error
            db2 = np.sum(error, axis=0, keepdims=True)
            da1 = error @ self.W2.T * (1 - self.a1**2)
            dW1 = X.T @ da1
            db1 = np.sum(da1, axis=0, keepdims=True)
            
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

# ===============================
# Simulate Data
# ===============================
np.random.seed(42)
num_chunks = 10
num_samples_per_chunk = 50
num_features = 20
num_classes = 3

data_chunks = [np.random.rand(num_samples_per_chunk, num_features) for _ in range(num_chunks)]
label_chunks = [np.random.randint(0, num_classes, num_samples_per_chunk) for _ in range(num_chunks)]

# ===============================
# Initialize Model
# ===============================
X_sample = non_logic_boost(data_chunks[0])
model = OnlineMLP(input_dim=X_sample.shape[1], hidden_dim=32, output_dim=num_classes, lr=0.01)

# ===============================
# Benchmark Simulation
# ===============================
all_preds, all_labels = [], []

for i, (X_chunk, y_chunk) in enumerate(zip(data_chunks, label_chunks), 1):
    X_chunk_boosted = non_logic_boost(X_chunk)
    model.update(X_chunk_boosted, y_chunk, epochs=2)
    
    probs = model.forward(X_chunk_boosted)
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(y_chunk, preds)
    f1 = f1_score(y_chunk, preds, average='weighted')
    
    print(f"Chunk {i} | Acc={acc:.4f} | F1={f1:.4f}")
    
    all_preds.extend(preds)
    all_labels.extend(y_chunk)

total_acc = accuracy_score(all_labels, all_preds)
total_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\nTotal | Acc={total_acc:.4f} | F1={total_f1:.4f}")
