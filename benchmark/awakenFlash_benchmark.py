import os, time, gzip, io, requests
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ==============================
# ✅ CONFIG
# ==============================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
RESULT_DIR = "benchmark_results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ==============================
# ✅ STREAMING DATA LOADER
# ==============================
def stream_covtype(chunksize=50000):
    print(f"🔗 Loading dataset streaming from: {DATA_URL}")
    with requests.get(DATA_URL, stream=True) as r:
        r.raise_for_status()
        decompressed = gzip.GzipFile(fileobj=io.BytesIO(r.content))
        df_iter = pd.read_csv(decompressed, header=None, chunksize=chunksize)
        for chunk in df_iter:
            X = chunk.iloc[:, :-1].values
            y = chunk.iloc[:, -1].values
            yield X, y

# ==============================
# ✅ RLS (Recursive Least Squares)
# ==============================
class OneStepRLS:
    def __init__(self, n_features, lam=1.0):
        self.w = np.zeros((n_features, 1))
        self.P = np.eye(n_features) / lam

    def partial_fit(self, X, y):
        for xi, yi in zip(X, y):
            xi = xi.reshape(-1, 1)
            # ✅ แก้ warning จาก NumPy
            err = float(yi - (xi.T @ self.w).item())
            g = self.P @ xi / (1.0 + (xi.T @ self.P @ xi).item())
            self.w += g * err
            self.P -= g @ xi.T @ self.P

    def predict(self, X):
        return np.sign(X @ self.w).flatten()

# ==============================
# ✅ MAIN BENCHMARK
# ==============================
def benchmark():
    results = []
    chunk_id = 0

    scaler = StandardScaler()
    sgd = SGDClassifier(max_iter=5)
    xgb = XGBClassifier(
        objective="multi:softmax", num_class=8, eval_metric="mlogloss", use_label_encoder=False
    )
    rls = None

    print("🚀 Starting benchmark...")

    for X, y in stream_covtype(chunksize=50000):
        chunk_id += 1
        print(f"\n===== Processing Chunk {chunk_id:02d} =====")

        X = scaler.fit_transform(X)
        y = y - y.min()  # normalize labels to start from 0

        if rls is None:
            rls = OneStepRLS(n_features=X.shape[1])

        chunk_result = {"chunk": chunk_id}

        # --- SGD ---
        t0 = time.time()
        sgd.partial_fit(X, y, classes=np.unique(y))
        pred_sgd = sgd.predict(X)
        acc_sgd = accuracy_score(y, pred_sgd)
        chunk_result["SGD_acc"] = acc_sgd
        chunk_result["SGD_time"] = time.time() - t0
        print(f"SGD: acc={acc_sgd:.3f}, time={chunk_result['SGD_time']:.3f}s")

        # --- RLS ---
        t0 = time.time()
        rls.partial_fit(X, y)
        pred_rls = np.round((X @ rls.w).flatten())
        acc_rls = np.mean(pred_rls == y)
        chunk_result["RLS_acc"] = acc_rls
        chunk_result["RLS_time"] = time.time() - t0
        print(f"RLS: acc={acc_rls:.3f}, time={chunk_result['RLS_time']:.3f}s")

        # --- XGBoost ---
        t0 = time.time()
        xgb.fit(X, y)
        pred_xgb = xgb.predict(X)
        acc_xgb = accuracy_score(y, pred_xgb)
        chunk_result["XGB_acc"] = acc_xgb
        chunk_result["XGB_time"] = time.time() - t0
        print(f"XGB: acc={acc_xgb:.3f}, time={chunk_result['XGB_time']:.3f}s")

        results.append(chunk_result)

    # ==============================
    # ✅ สรุปผลและบันทึก CSV
    # ==============================
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULT_DIR, "raw_results.csv"), index=False)

    summary = pd.DataFrame({
        "Model": ["SGD", "RLS", "XGB"],
        "Avg_Accuracy": [
            df["SGD_acc"].mean(),
            df["RLS_acc"].mean(),
            df["XGB_acc"].mean(),
        ],
        "Avg_Time(s)": [
            df["SGD_time"].mean(),
            df["RLS_time"].mean(),
            df["XGB_time"].mean(),
        ],
    })

    summary.to_csv(os.path.join(RESULT_DIR, "summary.csv"), index=False)
    print("\n📊 Benchmark Summary:\n", summary)


if __name__ == "__main__":
    benchmark()
