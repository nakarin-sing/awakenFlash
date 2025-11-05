import numpy as np
import time
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import pairwise_kernels

# ----------------------------------------------------------------------
# 1. Optimized OneStep Kernel Classifier (RLS/LS-SVM)
# ----------------------------------------------------------------------

class OptimizedOneStepClassifier:
    """Optimized Batch OneStep Kernel Classifier (Kernel RLS/LS-SVM)"""
    def __init__(self, C=10.0, kernel='rbf', gamma=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.X_train = None
        self.alpha = None
        self.classes_ = None

    def _get_non_attachment_gamma(self, X):
        """Data-Driven Gamma (Non-Attachment Principle)"""
        if X.var() > 0:
            return 1.0 / X.var()
        return 1.0 / X.shape[1]

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Prepare Target Vector (y)
        if n_classes == 2:
            y_proc = np.where(y == self.classes_[0], -1, 1)
            y_proc = y_proc[:, np.newaxis] # Ensure it's 2D for consistent matrix op
        else:
            y_map = {cls: i for i, cls in enumerate(self.classes_)}
            y_int = np.array([y_map[label] for label in y])
            y_proc = np.eye(n_classes)[y_int]

        # Compute Kernel Matrix (K)
        if self.kernel == 'rbf' and self.gamma is None:
            self.gamma = self._get_non_attachment_gamma(X)

        K = pairwise_kernels(X, metric=self.kernel, gamma=self.gamma)
        
        # Regularization Term: A = K + (1/C) * I
        A = K + np.eye(n_samples) / self.C
        
        # PicoQuantumEmulator OneStep Solution: Use lstsq for stability
        self.alpha, _, _, _ = np.linalg.lstsq(A, y_proc, rcond=None)
        
        return self

    def predict(self, X):
        if self.alpha is None:
            raise ValueError("Model must be fitted before calling predict.")

        K_test = pairwise_kernels(X, self.X_train, metric=self.kernel, gamma=self.gamma)
        scores = K_test @ self.alpha
        
        if len(self.classes_) == 2:
            y_pred_mapped = np.sign(scores.flatten())
            y_pred = np.where(y_pred_mapped == 1, self.classes_[1], self.classes_[0])
        else:
            y_pred_int = np.argmax(scores, axis=1)
            y_pred = self.classes_[y_pred_int]
            
        return y_pred

# ----------------------------------------------------------------------
# 2. Benchmark Setup
# ----------------------------------------------------------------------

def run_benchmark(dataset_loader, dataset_name, test_size=0.3, C_val=10.0):
    """Runs the benchmark comparison for a single dataset."""
    print(f"==========================================")
    print(f"🔬 Testing: {dataset_name}")
    print(f"==========================================")
    
    # Load Data
    data = dataset_loader()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    results = {}

    # --- Test 1: Optimized OneStep Kernel Classifier ---
    model_os = OptimizedOneStepClassifier(C=C_val, kernel='rbf')
    start_time = time.time()
    model_os.fit(X_train, y_train)
    train_time_os = time.time() - start_time
    
    y_pred_os = model_os.predict(X_test)
    acc_os = accuracy_score(y_test, y_pred_os)
    
    results['OneStep'] = {'Accuracy': acc_os, 'Time': train_time_os}

    # --- Test 2: XGBoost Classifier ---
    # Use default settings for a fair out-of-the-box comparison
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    start_time = time.time()
    model_xgb.fit(X_train, y_train)
    train_time_xgb = time.time() - start_time
    
    y_pred_xgb = model_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)

    results['XGBoost'] = {'Accuracy': acc_xgb, 'Time': train_time_xgb}
    
    # --- Print Summary ---
    print(f"🚀 OneStep (Kernel RLS): Acc={acc_os:.4f}, Time={train_time_os:.4f}s")
    print(f"🌲 XGBoost: Acc={acc_xgb:.4f}, Time={train_time_xgb:.4f}s")
    
    if acc_os > acc_xgb and train_time_os < train_time_xgb:
        print("\n🏆 **OneStep WINNER**: Higher Accuracy AND Faster Training!")
    elif acc_os > acc_xgb:
        print("\n🥇 **OneStep WINNER**: Higher Accuracy!")
    elif acc_xgb > acc_os:
        print("\n🥇 **XGBoost WINNER**: Higher Accuracy!")
    else:
        print("\n🤝 **TIE**: Results are very close.")

    print("------------------------------------------")
    
    return results

# ----------------------------------------------------------------------
# 3. Execute Benchmarks
# ----------------------------------------------------------------------

# 1. Multiclass Classification (Small Data)
run_benchmark(load_iris, "Iris Dataset (Small Multiclass)")

# 2. Binary Classification (Mid-size Data)
run_benchmark(load_breast_cancer, "Breast Cancer Dataset (Binary)")

# 3. Multiclass Classification (Mid-size Data)
run_benchmark(load_wine, "Wine Dataset (Multiclass)")

