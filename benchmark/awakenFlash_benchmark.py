# =======================
# Breast Cancer Benchmark
# =======================
import numpy as np
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

# -----------------------
# Fix warnings & reproducibility
# -----------------------
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------
# Load dataset
# -----------------------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.63, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# Poly2 Wrapper
# -----------------------
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.model = LogisticRegression(C=self.C, max_iter=2000, solver='lbfgs', random_state=RANDOM_STATE)
        
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self
    
    def predict(self, X):
        return self.model.predict(self.poly.transform(X))
    
    def predict_proba(self, X):
        return self.model.predict_proba(self.poly.transform(X))
    
    def score(self, X, y):
        return self.model.score(self.poly.transform(X), y)

# -----------------------
# RFF Wrapper
# -----------------------
class RFFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=100, gamma=None):
        self.n_components = n_components
        self.gamma = gamma
        self.model = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=RANDOM_STATE)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.gamma is None:
            var = np.var(X)
            self.gamma = 1.0 / (n_features * var) if var > 0 else 1.0
        rng = np.random.default_rng(RANDOM_STATE)
        self.W = rng.normal(0, np.sqrt(2*self.gamma), size=(X.shape[1], self.n_components))
        self.b = rng.uniform(0, 2*np.pi, size=self.n_components)
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        self.model.fit(Z, y)
        return self
    
    def predict(self, X):
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        return self.model.predict(Z)
    
    def predict_proba(self, X):
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        return self.model.predict_proba(Z)
    
    def score(self, X, y):
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        return self.model.score(Z, y)

# -----------------------
# Models
# -----------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE
)

poly2 = Poly2Wrapper(C=1.0)
rff = RFFWrapper(n_components=100)

# -----------------------
# Ensemble
# -----------------------
ensemble = VotingClassifier(
    estimators=[('XGBoost', xgb_model), ('Poly2', poly2), ('RFF', rff)],
    voting='soft'
)

models = [('XGBoost', xgb_model), ('Poly2', poly2), ('RFF', rff), ('Ensemble', ensemble)]

# -----------------------
# Evaluation
# -----------------------
print(f"{'Model':<10} | {'Train ACC':<10} | {'Test ACC':<10} | {'F1':<10} | {'CV mean':<10}")
print("-"*65)
for name, model in models:
    model.fit(X_train_scaled, y_train)
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    # cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    
    print(f"{name:<10} | {train_acc:<10.4f} | {test_acc:<10.4f} | {f1:<10.4f} | {cv_mean:<10.4f}")
