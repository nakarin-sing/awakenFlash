import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# Fix random seed
RANDOM_STATE = 42

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.63, random_state=RANDOM_STATE, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Poly2Wrapper -----
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, degree=2, C=1.0):
        self.degree = degree
        self.C = C
        self.model = LogisticRegression(max_iter=5000, C=C, random_state=RANDOM_STATE)
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self
    
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# ----- RFFWrapper -----
class RFFWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, n_components=500, gamma=1.0, C=1.0):
        self.n_components = n_components
        self.gamma = gamma
        self.C = C
        self.model = LogisticRegression(max_iter=5000, C=C, random_state=RANDOM_STATE)
        self.W = None
        self.b = None
    
    def fit(self, X, y):
        np.random.seed(RANDOM_STATE)
        d = X.shape[1]
        self.W = np.random.normal(scale=np.sqrt(2*self.gamma), size=(d, self.n_components))
        self.b = np.random.uniform(0, 2*np.pi, size=self.n_components)
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        self.model.fit(Z, y)
        return self
    
    def transform(self, X):
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        return Z
    
    def predict(self, X):
        Z = self.transform(X)
        return self.model.predict(Z)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# ----- XGBoost -----
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=RANDOM_STATE
)

# ----- Ensemble -----
ensemble = VotingClassifier(
    estimators=[
        ("xgb", xgb_model),
        ("poly2", Poly2Wrapper()),
        ("rff", RFFWrapper())
    ],
    voting="soft"
)

# ----- Run benchmark -----
models = [xgb_model, Poly2Wrapper(), RFFWrapper(), ensemble]
results = []

for model in models:
    # Fit
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    # CV mean
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = np.mean(cv_scores)
    
    results.append({
        "Model": model.__class__.__name__,
        "Train ACC": train_acc,
        "Test ACC": test_acc,
        "F1": f1,
        "CV mean": cv_mean
    })

# Output table
df = pd.DataFrame(results)
print(df.to_string(index=False))
