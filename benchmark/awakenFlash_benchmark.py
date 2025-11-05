import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# === NonLogic Term ===
def NonLogic_n(x, n=5, alpha=0.7):
    x = np.asarray(x)
    x_n = alpha * (0.5 - np.abs(x - 0.5))**n + (1 - alpha) * (0.5 + (-1)**n * (x - 0.5) / (2**(n-1)))
    return x_n

# === HybridStreamer RLS ===
class HybridStreamerRLS:
    def __init__(self, n_features, n_classes, lambda_reg=1e-3, alpha_non=0.7, n_non=5):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lambda_reg = lambda_reg
        self.alpha_non = alpha_non
        self.n_non = n_non
        self.X_train = None
        self.Y_train = None
        self.alpha_ = None
        self.fitted = False

    def partial_fit(self, X_batch, y_batch, classes):
        y_batch_zero_based = y_batch - np.min(classes)
        Y_onehot = np.eye(len(classes))[y_batch_zero_based]

        if self.X_train is None:
            self.X_train = X_batch
            self.Y_train = Y_onehot
        else:
            self.X_train = np.vstack([self.X_train, X_batch])
            self.Y_train = np.vstack([self.Y_train, Y_onehot])

        I = np.eye(self.X_train.shape[0])
        K = self.X_train @ self.X_train.T
        self.alpha_ = np.linalg.solve(K + self.lambda_reg * I, self.Y_train)
        self.fitted = True

    def predict_proba(self, X_test):
        if not self.fitted:
            raise ValueError("Model not fitted yet!")
        pred_raw = X_test @ self.X_train.T @ self.alpha_
        pred_min = pred_raw.min(axis=0)
        pred_max = pred_raw.max(axis=0)
        pred_norm = (pred_raw - pred_min) / (pred_max - pred_min + 1e-12)
        pred_non = NonLogic_n(pred_norm, n=self.n_non, alpha=self.alpha_non)
        pred_non /= pred_non.sum(axis=1, keepdims=True)
        return pred_non

    def predict(self, X_test):
        pred_non = self.predict_proba(X_test)
        pred_labels = np.argmax(pred_non, axis=1) + 1
        return pred_labels

# === Load Data ===
data = fetch_covtype()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Streaming Setup ===
chunk_size = 10000
n_chunks = X_train.shape[0] // chunk_size

hybrid_streamer = HybridStreamerRLS(n_features=X.shape[1], n_classes=7, alpha_non=0.8, n_non=5)
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss'
)

# === Streaming Simulation ===
hybrid_acc = []
xgb_acc = []

for i in range(n_chunks):
    X_batch = X_train[i*chunk_size:(i+1)*chunk_size]
    y_batch = y_train[i*chunk_size:(i+1)*chunk_size]

    # HybridStreamer update
    hybrid_streamer.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
    y_pred_hybrid = hybrid_streamer.predict(X_test)
    hybrid_acc.append(np.mean(y_pred_hybrid == y_test))

    # XGBoost incremental (fit on chunk)
    xgb_model.fit(X_batch, y_batch)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_acc.append(np.mean(y_pred_xgb == y_test))

    print(f"Chunk {i+1}: HybridStreamer Acc={hybrid_acc[-1]:.4f} | XGBoost Acc={xgb_acc[-1]:.4f}")

# === Summary ===
print("\n=== Streaming Benchmark Summary ===")
print(f"HybridStreamer Average Acc: {np.mean(hybrid_acc):.4f}")
print(f"XGBoost Average Acc: {np.mean(xgb_acc):.4f}")
