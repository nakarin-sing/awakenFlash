import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

# -----------------------------
# Wrappers
# -----------------------------
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, degree=2, C=1.0):
        self.degree = degree
        self.C = C
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.model = LogisticRegression(max_iter=5000, C=self.C, random_state=42)
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

class RFFWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"
    def __init__(self, gamma=1.0, n_components=500, C=1.0):
        self.gamma = gamma
        self.n_components = n_components
        self.C = C
        self.rbf = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=42)
        self.model = LogisticRegression(max_iter=5000, C=self.C, random_state=42)
    def fit(self, X, y):
        X_rbf = self.rbf.fit_transform(X)
        self.model.fit(X_rbf, y)
        return self
    def predict(self, X):
        X_rbf = self.rbf.transform(X)
        return self.model.predict(X_rbf)
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# -----------------------------
# Load dataset
# -----------------------------
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.63, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Models
# -----------------------------
models = {
    "XGBoost": xgb.XGBClassifier(
        max_depth=3, n_estimators=200, use_label_encoder=False,
        eval_metric='logloss', random_state=42
    ),
    "Poly2": Poly2Wrapper(degree=2, C=1.0),
    "RFF": RFFWrapper(gamma=0.1, n_components=500, C=1.0)
}

# -----------------------------
# Benchmark
# -----------------------------
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cv_mean = cross_val_score(model, X_train_scaled, y_train, cv=5).mean()
    results.append({
        "Model": name,
        "Train ACC": f"{train_acc:.4f}",
        "Test ACC": f"{test_acc:.4f}",
        "F1": f"{f1:.4f}",
        "CV mean": f"{cv_mean:.4f}"
    })

# -----------------------------
# Voting Ensemble
# -----------------------------
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='hard'
)
ensemble.fit(X_train_scaled, y_train)
y_pred = ensemble.predict(X_test_scaled)
results.append({
    "Model": "Ensemble",
    "Train ACC": f"{ensemble.score(X_train_scaled, y_train):.4f}",
    "Test ACC": f"{accuracy_score(y_test, y_pred):.4f}",
    "F1": f"{f1_score(y_test, y_pred):.4f}",
    "CV mean": f"{cross_val_score(ensemble, X_train_scaled, y_train, cv=5).mean():.4f}"
})

# -----------------------------
# Display results
# -----------------------------
df = pd.DataFrame(results)
print(df.to_string(index=False))
