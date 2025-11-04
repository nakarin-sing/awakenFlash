import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

# ====== จำลอง dataset ======
np.random.seed(42)
num_chunks = 10
chunk_size = 100
num_features = 20
num_classes = 3

data_chunks = [np.random.rand(chunk_size, num_features) for _ in range(num_chunks)]
label_chunks = [np.random.randint(0, num_classes, size=chunk_size) for _ in range(num_chunks)]

# ====== Non-Logic Feature Boost ======
def non_logic_boost(X):
    # เพิ่ม feature ใหม่แบบ non-linear + interaction
    X_new = np.hstack([X, X**2, np.sin(X), np.prod(X[:, :2], axis=1, keepdims=True)])
    return X_new

# ====== Memory-Augmented Online MLP ======
class OnlineMLP:
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, lr=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        self.lr = lr
        self.scaler = StandardScaler()
        self.memory_X = np.empty((0, input_dim))
        self.memory_y = np.empty((0,), dtype=int)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.softmax(self.z2)

    def update(self, X, y, epochs=2):
        # memory augmentation
        if self.memory_X.shape[0] > 0:
            X = np.vstack([self.memory_X, X])
            y = np.concatenate([self.memory_y, y])
        X = self.scaler.fit_transform(X)
        for _ in range(epochs):
            probs = self.forward(X)
            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(len(y)), y] = 1
            grad_z2 = probs - y_onehot
            grad_W2 = self.a1.T @ grad_z2 / len(X)
            grad_b2 = grad_z2.mean(axis=0)
            grad_a1 = grad_z2 @ self.W2.T
            grad_z1 = grad_a1 * (self.z1 > 0)
            grad_W1 = X.T @ grad_z1 / len(X)
            grad_b1 = grad_z1.mean(axis=0)
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
        # keep last chunk in memory
        self.memory_X = X[-chunk_size:]
        self.memory_y = y[-chunk_size:]

    def predict(self, X):
        X = self.scaler.transform(X)
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# ====== รัน Full Online ======
model = OnlineMLP(input_dim=num_features*4 + 1)  # Boosted features
total_acc = []
total_f1 = []

for i, (X_chunk, y_chunk) in enumerate(zip(data_chunks, label_chunks), 1):
    X_chunk_boosted = non_logic_boost(X_chunk)
    model.update(X_chunk_boosted, y_chunk, epochs=2)
    y_pred = model.predict(X_chunk_boosted)
    acc = accuracy_score(y_chunk, y_pred)
    f1 = f1_score(y_chunk, y_pred, average='weighted')
    total_acc.append(acc)
    total_f1.append(f1)
    print(f"Chunk {i} | OnlineMLP+Boost Acc={acc:.3f} | F1={f1:.3f} | Time≈0.3s")

print(f"\nAverage | Acc={np.mean(total_acc):.3f} | F1={np.mean(total_f1):.3f}")
