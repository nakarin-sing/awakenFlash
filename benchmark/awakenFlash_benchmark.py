import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

# -----------------------------
# Dummy Dataset
# -----------------------------
X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=15,
    n_classes=7, random_state=42
)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Remap labels to 0-based for XGBoost
classes = np.unique(y_train)
y_train_zero = y_train - classes.min()
y_test_zero = y_test - classes.min()

# -----------------------------
# HybridStreamerRLS placeholder
# -----------------------------
class HybridStreamerRLS:
    def __init__(self, n_features, n_classes, reg_lambda=1e-3):
        self.n_features = n_features
        self.n_classes = n_classes
        self.reg_lambda = reg_lambda
        self.W = np.zeros((n_features, n_classes))
    
    def partial_fit(self, X_batch, y_batch, classes):
        # One-hot encode
        Y_onehot = np.eye(len(classes))[y_batch]
        # Closed-form RLS update
        XtX = X_batch.T @ X_batch
        self.W = np.linalg.solve(XtX + self.reg_lambda*np.eye(XtX.shape[0]), X_batch.T @ Y_onehot)
    
    def predict(self, X):
        scores = X @ self.W
        return np.argmax(scores, axis=1)

# -----------------------------
# Streaming Benchmark
# -----------------------------
def streaming_benchmark(X_train, y_train_zero, X_test, y_test_zero, batch_size=1000):
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train_zero))
    streamer = HybridStreamerRLS(n_features, n_classes)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    
    n_batches = X_train.shape[0] // batch_size
    for i in range(n_batches):
        start = i*batch_size
        end = (i+1)*batch_size
        X_batch = X_train[start:end]
        y_batch = y_train_zero[start:end]
        
        # Partial fit HybridStreamerRLS
        streamer.partial_fit(X_batch, y_batch, classes=range(n_classes))
        
        # Fit XGBoost batch
        xgb_model.fit(X_batch, y_batch)
    
    # Predict
    y_pred_streamer = streamer.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    
    acc_streamer = np.mean(y_pred_streamer == y_test_zero)
    acc_xgb = np.mean(y_pred_xgb == y_test_zero)
    
    print(f"HybridStreamerRLS Accuracy: {acc_streamer:.4f}")
    print(f"XGBoost Accuracy:          {acc_xgb:.4f}")

# -----------------------------
# Run benchmark
# -----------------------------
streaming_benchmark(X_train, y_train_zero, X_test, y_test_zero, batch_size=1000)
