import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import RBFSampler

# ------------------------------
# Wrappers
# ------------------------------
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.model = LogisticRegression(C=self.C, max_iter=5000)
    
    def fit(self, X, y):
        from sklearn.preprocessing import PolynomialFeatures
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self
    
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)
    
    def predict_proba(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict_proba(X_poly)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class RFFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=1.0, n_components=100):
        self.gamma = gamma
        self.n_components = n_components
        self.model = LogisticRegression(max_iter=5000)
    
    def fit(self, X, y):
        self.rff = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=42)
        X_rff = self.rff.fit_transform(X)
        self.model.fit(X_rff, y)
        return self
    
    def predict(self, X):
        X_rff = self.rff.transform(X)
        return self.model.predict(X_rff)
    
    def predict_proba(self, X):
        X_rff = self.rff.transform(X)
        return self.model.predict_proba(X_rff)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# ------------------------------
# Load Data
# ------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.63, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# Models
# ------------------------------
xgb_model = GradientBoostingClassifier(random_state=42)
poly_model = Poly2Wrapper(C=1.0)
rff_model = RFFWrapper(gamma=1.0 / (X_train_scaled.shape[1]*np.var(X_train_scaled)), n_components=min(100, X_train_scaled.shape[0]//2))

ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('poly', poly_model),
        ('rff', rff_model)
    ],
    voting='soft'
)

models = [xgb_model, poly_model, rff_model, ensemble]
names = ['XGBoost', 'Poly2', 'RFF', 'Ensemble']

# ------------------------------
# Benchmark
# ------------------------------
print(f"{'Model':<10} | {'Train ACC':<10} | {'Test ACC':<10} | {'F1':<10} | {'CV mean':<10}")
for model, name in zip(models, names):
    model.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    f1 = f1_score(y_test, model.predict(X_test_scaled))
    cv_mean = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    print(f"{name:<10} | {train_acc:<10.4f} | {test_acc:<10.4f} | {f1:<10.4f} | {cv_mean:<10.4f}")
