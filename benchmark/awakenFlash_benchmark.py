import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# -------------------
# Metrics helper
# -------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


# -------------------
# Load UCI Covertype dataset
# -------------------
def load_data(n_chunks=10, chunk_size=10000):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    print("📊 Loading dataset...")
    df = pd.read_csv(url, header=None)
    X_all = df.iloc[:, :-1].values
    y_all = df.iloc[:, -1].values - 1

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    chunks = [(X_all[i:i+chunk_size], y_all[i:i+chunk_size]) 
              for i in range(0, min(len(X_all), n_chunks * chunk_size), chunk_size)]
    return chunks, np.unique(y_all)


# -------------------
# Simple ensemble (Temporal)
# -------------------
class SimpleTemporal:
    def __init__(self):
        self.models = [
            SGDClassifier(loss='log_loss', max_iter=10, warm_start=True, random_state=42),
            PassiveAggressiveClassifier(max_iter=10, warm_start=True, random_state=42)
        ]
        self.classes_ = None

    def partial_fit(self, X, y, classes=None):
        if self.classes_ is None and classes is not None:
            self.classes_ = classes
        for m in self.models:
            if classes is not None:
                m.partial_fit(X, y, classes=classes)
            else:
                m.partial_fit(X, y)

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes_)))
        for m in self.models:
            pred = m.predict(X)
            for i, cls in enumerate(self.classes_):
                votes[:, i] += (pred == cls)
        return self.classes_[np.argmax(votes, axis=1)]


# -------------------
# Benchmark runner
# -------------------
def run_benchmark():
    chunks, all_classes = load_data(n_chunks=10, chunk_size=10000)

    temporal = SimpleTemporal()
    sgd = SGDClassifier(loss='log_loss', max_iter=10, warm_start=True, random_state=42)
    pa = PassiveAggressiveClassifier(max_iter=10, warm_start=True, random_state=42)

    xgb_all_X, xgb_all_y = [], []
    WINDOW_SIZE = 5

    first_sgd = first_pa = first_temporal = True
    results = []

    for idx, (X_chunk, y_chunk) in enumerate(chunks, 1):
        split = int(0.8 * len(X_chunk))
        X_train, X_test = X_chunk[:split], X_chunk[split:]
        y_train, y_test = y_chunk[:split], y_chunk[split:]

        # Temporal
        start = time.time()
        if first_temporal:
            temporal.partial_fit(X_train, y_train, classes=all_classes)
            first_temporal = False
        else:
            temporal.partial_fit(X_train, y_train)
        temporal_pred = temporal.predict(X_test)
        temporal_metrics = compute_metrics(y_test, temporal_pred)
        temporal_time = time.time() - start

        # SGD
        start = time.time()
        if first_sgd:
            sgd.partial_fit(X_train, y_train, classes=all_classes)
            first_sgd = False
        else:
            sgd.partial_fit(X_train, y_train)
        sgd_pred = sgd.predict(X_test)
        sgd_metrics = compute_metrics(y_test, sgd_pred)
        sgd_time = time.time() - start

        # PA
        start = time.time()
        if first_pa:
            pa.partial_fit(X_train, y_train, classes=all_classes)
            first_pa = False
        else:
            pa.partial_fit(X_train, y_train)
        pa_pred = pa.predict(X_test)
        pa_metrics = compute_metrics(y_test, pa_pred)
        pa_time = time.time() - start

        # XGBoost
        start = time.time()
        xgb_all_X.append(X_train)
        xgb_all_y.append(y_train)
        if len(xgb_all_X) > WINDOW_SIZE:
            xgb_all_X = xgb_all_X[-WINDOW_SIZE:]
            xgb_all_y = xgb_all_y[-WINDOW_SIZE:]

        X_xgb = np.vstack(xgb_all_X)
        y_xgb = np.concatenate(xgb_all_y)

        dtrain = xgb.DMatrix(X_xgb, label=y_xgb)
        dtest = xgb.DMatrix(X_test, label=y_test)

        xgb_model = xgb.train(
            {"objective": "multi:softmax", "num_class": len(all_classes), "max_depth": 5,
             "eta": 0.1, "subsample": 0.8, "verbosity": 0},
            dtrain, num_boost_round=20
        )
        xgb_pred = xgb_model.predict(dtest)
        xgb_metrics = compute_metrics(y_test, xgb_pred)
        xgb_time = time.time() - start

        # Store
        results.append({
            'chunk': idx,
            'temporal_acc': temporal_metrics['accuracy'],
            'temporal_f1': temporal_metrics['f1'],
            'temporal_time': temporal_time,
            'sgd_acc': sgd_metrics['accuracy'],
            'sgd_f1': sgd_metrics['f1'],
            'sgd_time': sgd_time,
            'pa_acc': pa_metrics['accuracy'],
            'pa_f1': pa_metrics['f1'],
            'pa_time': pa_time,
            'xgb_acc': xgb_metrics['accuracy'],
            'xgb_f1': xgb_metrics['f1'],
            'xgb_time': xgb_time
        })

        # Print log per chunk
        print(f"Chunk {idx:02d}: Temporal={temporal_metrics['accuracy']:.3f} | "
              f"XGB={xgb_metrics['accuracy']:.3f} | "
              f"Time Temporal={temporal_time:.3f}s, XGB={xgb_time:.3f}s")

    df_results = pd.DataFrame(results)
    os.makedirs('benchmark_results', exist_ok=True)
    df_results.to_csv('benchmark_results/fast_non_dualistic_results.csv', index=False)
    print("\n📄 Results saved to benchmark_results/fast_non_dualistic_results.csv")

    return df_results


if __name__ == "__main__":
    run_benchmark()
