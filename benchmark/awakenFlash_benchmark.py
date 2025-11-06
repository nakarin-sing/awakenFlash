import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score
import psutil
import os

def cpu_time():
    p = psutil.Process(os.getpid())
    return p.cpu_times().user + p.cpu_times().system

class OneStepNystrom(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, n_components=1000, gamma='scale', random_state=42):
        self.C = C
        self.n_components = n_components  # m = จำนวน landmark
        self.gamma = gamma
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.landmarks_ = None
        self.alpha_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = self.scaler.fit_transform(X).astype(np.float32)
        n_samples = X.shape[0]
        
        # 1. เลือก landmark แบบสุ่ม (m << n)
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n_samples)[:self.n_components]
        self.landmarks_ = X[idx]
        
        # 2. คำนวณ K_nm = X @ landmarks_.T
        if self.gamma == 'scale':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma
            
        K_nm = np.exp(-gamma * ((X ** 2).sum(axis=1)[:, None] + 
                                (self.landmarks_ ** 2).sum(axis=1)[None, :] - 
                                2 * X @ self.landmarks_.T))
        
        # 3. คำนวณ K_mm = landmarks_ @ landmarks_.T
        K_mm = np.exp(-gamma * ((self.landmarks_ ** 2).sum(axis=1)[:, None] + 
                                (self.landmarks_ ** 2).sum(axis=1)[None, :] - 
                                2 * self.landmarks_ @ self.landmarks_.T))
        
        # 4. Regularize และ solve: alpha = K_nm @ (K_mm + λI)^(-1) y
        lambda_reg = self.C * np.trace(K_mm) / self.n_components
        K_reg = K_mm + lambda_reg * np.eye(self.n_components, dtype=np.float32)
        
        # One-hot encode y
        self.classes_ = np.unique(y)
        y_onehot = np.zeros((len(y), len(self.classes_)), dtype=np.float32)
        for i, cls in enumerate(self.classes_):
            y_onehot[y == cls, i] = 1.0
        
        # Solve small system: (K_mm + λI) β = K_mn^T y → β = solve(...)
        beta = np.linalg.solve(K_reg, K_nm.T @ y_onehot)
        self.alpha_ = K_nm @ beta  # n × n_classes
        
        return self

    def predict(self, X):
        X = self.scaler.transform(X).astype(np.float32)
        if self.gamma == 'scale':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma
            
        K_test = np.exp(-gamma * ((X ** 2).sum(axis=1)[:, None] + 
                                  (self.landmarks_ ** 2).sum(axis=1)[None, :] - 
                                  2 * X @ self.landmarks_.T))
        scores = K_test @ self.alpha_
        return self.classes_[np.argmax(scores, axis=1)]
