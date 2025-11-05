import numpy as np
import time
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer

# ======================================================================
# 1. Definitive OneStep Kernel Classifier (Kernel RLS/LS-SVM)
#    - Optimized for Multi-Class Stability (using LabelBinarizer)
#    - Utilizes lstsq for PicoQuantumEmulator Stability
# ======================================================================

class DefinitiveOneStepClassifier:
    """
    Definitive OneStep Classifier aiming for UNDISPUTED VICTORY over XGBoost 
    by optimizing Multi-Class stability and efficiency.
    """
    
    def __init__(self, C=10.0, kernel='rbf', gamma=None):
        self.C = C 
        self.kernel = kernel
        self.gamma = gamma
        self.X_train = None
        self.alpha = None
        self.classes_ = None
        self.lb = None

    def _get_non_attachment_gamma(self, X):
        """Data-Driven Gamma (Non-Attachment Principle)"""
        # More robust heuristic: 1 / variance of the data
        if X.var() > 0:
            return 1.0 / X.var()
        return 1.0 / X.shape[1]

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        
        # 1. Prepare Target Matrix (Y) - Multi-Output Binarization
        self.lb = LabelBinarizer()
        y_bin = self.lb.fit_transform(y)
        # Ensure Y is 2D (n_samples, n_classes)
        Y = y_bin if y_bin.ndim > 1 else y_bin[:, np.newaxis] 

        # 2. Compute Kernel Matrix (K)
        if self.kernel == 'rbf' and self.gamma is None:
            self.gamma = self._get_non_attachment_gamma(X)

        K = pairwise_kernels(X, metric=self.kernel, gamma=self.gamma)
        
        # 3. Regularization Term: A = K + (1/C) * I
        A = K + np.eye(n_samples) / self.C
        
        # 4. Optimized Closed-Form Solution (PicoQuantumEmulator Stability)
        # Solves: A @ alpha = Y
        self.alpha, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        
        return self

    def predict(self, X):
        if self.alpha is None:
            raise ValueError("Model must be fitted before calling predict.")

        # Compute K_test
        K_test = pairwise_kernels(X, self.X_train, metric=self.kernel, gamma=self.gamma)
        
        # Prediction Scores: Scores = K_test @ alpha 
        scores = K_test @ self.alpha
        
        # Inverse transform to get predicted class labels
        if scores.shape[1] == 1:
            # Binary case: Scores > 0.5 maps to class 1, otherwise class 0
            y_pred_bin = (scores > 0.5).astype(int).flatten()
            y_pred = self.lb.inverse_transform(y_pred_bin)
        else:
            # Multiclass: Argmax (index of the highest score is the predicted class)
            y_pred = self.lb.inverse_transform(scores)
            
        return y_pred

# ======================================================================
# 2. Benchmark Execution Logic
# ======================================================================

def run_benchmark(dataset_loader, dataset_name, test_size=0.3, C_val=10.0):
    """Runs the benchmark comparison for a single dataset and prints results."""
    print(f"\n==========================================")
    print(f"🔬 Testing: {dataset_name}")
    print(f"==========================================")
    
    # Load Data
    data = dataset_loader()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # --- Test 1: Definitive OneStep Kernel Classifier ---
    model_os = DefinitiveOneStepClassifier(C=C_val, kernel='rbf')
    start_time = time.time()
    model_os.fit(X_train, y_train)
    train_time_os = time.time() - start_time
    
    y_pred_os = model_os.predict(X_test)
    acc_os = accuracy_score(y_test, y_pred_os)
    
    # --- Test 2: XGBoost Classifier ---
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    start_time = time.time()
    model_xgb.fit(X_train, y_train)
    train_time_xgb = time.time() - start_time
    
    y_pred_xgb = model_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    
    # --- Print Summary (Crucial for CI Log) ---
    print(f"🚀 OneStep (Definitive RLS): Acc={acc_os:.4f}, Time={train_time_os:.4f}s")
    print(f"🌲 XGBoost: Acc={acc_xgb:.4f}, Time={train_time_xgb:.4f}s")
    
    if acc_os > acc_xgb and train_time_os < train_time_xgb:
        print("\n🏆 **OneStep UNDISPUTED CHAMPION**: Higher Accuracy AND Faster Training!")
    elif acc_os >= acc_xgb and train_time_os < train_time_xgb:
        print("\n🥇 **OneStep WINNER**: Equal/Higher Accuracy AND Faster Training!")
    elif acc_xgb > acc_os:
        print("\n🥇 **XGBoost WINNER**: Higher Accuracy!")
    else:
        print("\n🤝 **TIE**: Results are very close.")

    print("------------------------------------------")

# ======================================================================
# 3. Main Execution
# ======================================================================

if __name__ == "__main__":
    # Ensure all dependencies are installed before running (numpy, scikit-learn, xgboost)
    
    # 1. Multiclass Classification (Small Data)
    run_benchmark(load_iris, "Iris Dataset (Small Multiclass)", C_val=100.0)

    # 2. Binary Classification (Mid-size Data)
    run_benchmark(load_breast_cancer, "Breast Cancer Dataset (Binary)", C_val=100.0)

    # 3. Multiclass Classification (Mid-size Data)
    run_benchmark(load_wine, "Wine Dataset (Multiclass)", C_val=10.0)
    
    # Final message for the CI log
    print("\n✅ Benchmark Complete: Results logged above.")
