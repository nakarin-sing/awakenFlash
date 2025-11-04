import os, time, gzip, io, requests
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# ===============================================================
# 🧠 RLS++ — Recursive Least Squares with Nonlinear Expansion
# ===============================================================
class RLSPlus:
    def __init__(self, n_features, lam=0.98, delta=1.0):
        self.lam = lam
        self.delta = delta
        self.P = (1.0 / delta) * np.eye(n_features)
        self.w = np.zeros((n_features, 1))

    def expand_features(self, X):
        # Fourier + Polynomial Expansion (lightweight)
        X2 = X ** 2
        Xsin = np.sin(X * 0.5)
        Xcos = np.cos(X * 0.5)
        return np.hstack([X, X2, Xsin, Xcos])

    def update(self, X, y):
        X = self.expand_features(X)
        for xi, yi in zip(X, y):
            xi = xi.reshape(-1, 1)
            k = self.P @ xi / (self.lam + xi.T @ self.P @ xi)
            err = yi - float(xi.T @ self.w)
            self.w += k * err
            self.P = (self.P - k @ xi.T @ self.P) / self.lam
        # Adaptive lambda decay
        self.lam = max(0.95 * self.lam, 0.90)

    def predict(self, X):
        X = self.expand_features(X)
        preds = X @ self.w
        return (preds > 0.5).astype(int).ravel()


# ===============================================================
# 📊 Dataset Loader — UCI Covertype (Streaming)
# ===============================================================
def stream_covtype(url, chunk_size=10000):
    print(f"🔗 Loading dataset streaming from: [{url}]")
    with requests.get(url, stream=True) as r:
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as f:
            data = np.loadtxt(f, delimiter=",")
            X, y = data[:, :-1], data[:, -1]
            for i in range(0, len(X), chunk_size):
                yield X[i:i+chunk_size], y[i:i+chunk_size]


# ===============================================================
# 🚀 Benchmark Runner
# ===============================================================
def benchmark():
    os.makedirs("benchmark_results", exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    chunk_gen = stream_covtype(url)

    sgd = SGDClassifier(loss="log_loss", max_iter=1, learning_rate="optimal")
    xgb = XGBClassifier(
        n_estimators=20, max_depth=5, learning_rate=0.2,
        subsample=0.5, colsample_bytree=0.5, verbosity=0
    )

    scaler = StandardScaler()
    first_chunk = True
    chunk_logs = []

    print("🚀 Starting benchmark...\n")

    for i, (X, y) in enumerate(chunk_gen, 1):
        if i > 5:
            break  # Limit for CI speed

        X = scaler.fit_transform(X)
        y = (y > 3).astype(int)  # binary simplify

        # Initialize RLS++ with expanded feature dim
        if first_chunk:
            n_features_expanded = X.shape[1] * 4  # after expansion
            rls = RLSPlus(n_features_expanded)
            first_chunk = False

        # === SGD ===
        t0 = time.time()
        sgd.partial_fit(X, y, classes=np.unique(y))
        sgd_acc = accuracy_score(y, sgd.predict(X))
        sgd_time = time.time() - t0

        # === RLS++ ===
        t0 = time.time()
        rls.update(X, y)
        rls_acc = accuracy_score(y, rls.predict(X))
        rls_time = time.time() - t0

        # === XGB ===
        t0 = time.time()
        xgb.fit(X, y)
        xgb_acc = accuracy_score(y, xgb.predict(X))
        xgb_time = time.time() - t0

        msg = (
            f"===== Processing Chunk {i:02d} =====\n"
            f"SGD: acc={sgd_acc:.3f}, time={sgd_time:.3f}s\n"
            f"RLS++: acc={rls_acc:.3f}, time={rls_time:.3f}s\n"
            f"XGB: acc={xgb_acc:.3f}, time={xgb_time:.3f}s\n"
        )
        print(msg)
        chunk_logs.append(msg)

    with open("benchmark_results/chunk_log.txt", "w") as f:
        f.writelines(chunk_logs)

    print("✅ Benchmark finished successfully.")
    print("Results saved to benchmark_results/chunk_log.txt")


if __name__ == "__main__":
    benchmark()
