import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer

class DefinitiveOneStepClassifier:
    """
    Definitive OneStep Classifier aiming for UNDISPUTED VICTORY over XGBoost 
    by optimizing Multi-Class Kernel RLS stability.
    """
    
    def __init__(self, C=10.0, kernel='rbf', gamma=None):
        self.C = C 
        self.kernel = kernel
        self.gamma = gamma
        self.X_train = None
        self.alpha = None
        self.classes_ = None
        self.lb = None  # Label Binarizer for Multiclass Y handling

    def _get_non_attachment_gamma(self, X):
        """Principle: Non-Attachment Gamma Heuristic"""
        if X.var() > 0:
            return 1.0 / X.var()
        return 1.0 / X.shape[1]

    def fit(self, X, y):
        self.X_train = X
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        
        # 1. Prepare Target Matrix (Y) - Multi-Output Binarization (Optimal for RLS)
        self.lb = LabelBinarizer()
        y_bin = self.lb.fit_transform(y)
        # Ensure y_bin is 2D even for binary case (n_samples, 1)
        Y = y_bin if y_bin.ndim > 1 else y_bin[:, np.newaxis] 

        # 2. Compute Kernel Matrix (K)
        if self.kernel == 'rbf' and self.gamma is None:
            self.gamma = self._get_non_attachment_gamma(X)

        K = pairwise_kernels(X, metric=self.kernel, gamma=self.gamma)
        
        # 3. Regularization Term (A = K + (1/C) * I)
        # We solve the system for each output simultaneously (Multi-Output RLS)
        A = K + np.eye(n_samples) / self.C
        
        # 4. Optimized Closed-Form Solution (PicoQuantumEmulator Stability)
        # Solves: A @ alpha = Y  => alpha = lstsq(A, Y)
        # alpha will be (n_samples, n_classes)
        self.alpha, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        
        return self

    def predict(self, X):
        if self.alpha is None:
            raise ValueError("Model must be fitted before calling predict.")

        # Compute K_test
        K_test = pairwise_kernels(X, self.X_train, metric=self.kernel, gamma=self.gamma)
        
        # Prediction Scores: Scores = K_test @ alpha (n_test_samples, n_classes)
        scores = K_test @ self.alpha
        
        # Inverse transform to get predicted class labels
        if len(self.classes_) == 2:
            # For binary (1 column output), use threshold 0.5 for class 1, otherwise class 0
            y_pred_bin = (scores > 0.5).astype(int)
        else:
            # For multiclass, argmax to find the highest score (the winning class)
            y_pred_bin = np.argmax(scores, axis=1)
            
        # Map back to original labels (0, 1, 2, 'setosa', etc.)
        y_pred = self.lb.inverse_transform(y_pred_bin)
            
        return y_pred
