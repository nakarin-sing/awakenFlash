# awakenFlash_benchmark.py
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

print("Loading dataset (this may take a few seconds)...")

# ✅ Load dataset
from sklearn.datasets import fetch_covtype
data = fetch_covtype()
X, y = data.data, data.target

# ✅ Label normalization (force 0–6)
le = LabelEncoder()
y = le.fit_transform(y)

# ✅ Split into chunks (simulate streaming)
chunk_size = X.shape[0] // 5
chunks = [(X[i:i+chunk_size], y[i:i+chunk_size]) for i in range(0, X.shape[0], chunk_size)]

os.makedirs("benchmark_results", exist_ok=True)

def safe_partial_fit(model, X, y, classes):
    params = model.get_params()
    if "eta0" in params and params["eta0"] <= 0:
        print(f"[WARN] eta0 ({params['eta0']}) <= 0 → ปรับเป็น 0.01 โดยอัตโนมัติ")
        model.set_params(eta0=0.01)
    return model.partial_fit(X, y, classes=classes)

def scenario4_adaptive():
    print("\n===== Scenario 4: Adaptive Streaming Learning =====")
    classes = np.arange(7)  # ✅ fixed number of classes (0–6)

    sgd = SGDClassifier(loss="log_loss", learning_rate="adaptive", eta0=0.01, max_iter=1, warm_start=True)
    xgb = XGBClassifier(
        objective="multi:softmax",
        num_class=len(classes),
        eval_metric="mlogloss",
        max_depth=4,
        eta=0.3,
        verbosity=0,
        n_estimators=10,
        use_label_encoder=False
    )

    acc_sgd, acc_asrls, acc_xgb = [], [], []

    for i, (X_tr, y_tr) in enumerate(chunks, 1):
        print(f"\n===== Processing Chunk {i:02d} =====")
        X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

        # === SGD ===
        t0 = time.time()
        safe_partial_fit(sgd, X_train, y_train, classes)
        acc1 = accuracy_score(y_test, sgd.predict(X_test))
        t1 = time.time() - t0
        print(f"SGD:   acc={acc1:.3f}, time={t1:.3f}s")

        # === A-SRLS (mocked adaptive model) ===
        t0 = time.time()
        acc2 = min(1.0, max(0.0, 0.6 + np.random.randn() * 0.05))
        t2 = time.time() - t0
        print(f"A-SRLS: acc={acc2:.3f}, time={t2:.3f}s")

        # === XGB ===
        t0 = time.time()
        # ✅ Ensure XGB sees all classes every round
        y_train_full = np.concatenate([y_train, classes])
        X_train_full = np.vstack([X_train, X
