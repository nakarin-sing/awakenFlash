from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import xgboost as xgb

# ==============================
# Poly2Wrapper (Fixed)
# ==============================
from sklearn.preprocessing import PolynomialFeatures

class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.model = LogisticRegression(C=self.C, max_iter=1000)

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

# ==============================
# RFFWrapper (Simplified Realistic)
# ==============================
class RFFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=1.0, n_components=100, C=1.0):
        self.gamma = gamma
        self.n_components = n_components
        self.C = C
        self.model = LogisticRegression(C=self.C, max_iter=1000)

    def fit(self, X, y):
        D = X.shape[1]
        self.W = np.random.normal(0, np.sqrt(2*self.gamma), size=(D, self.n_components))
        self.b = np.random.uniform(0, 2*np.pi, size=self.n_components)
        Z = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        self.model.fit(Z, y)
        return self

    def transform(self, X):
        return np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)

    def predict(self, X):
        return self.model.predict(self.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))

    def score(self, X, y):
        return self.model.score(self.transform(X), y)

# ==============================
# Load Data
# ==============================
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.63, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Initialize Models
# ==============================
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
poly_model = Poly2Wrapper(C=1.0)
rff_model = RFFClassifier(gamma=0.1, n_components=100, C=1.0)

# ==============================
# Weighted Voting Ensemble
# ==============================
# weight by validation accuracy
weights = []
for m in [xgb_model, poly_model, rff_model]:
    m.fit(X_train_scaled, y_train)
    val_acc = m.score(X_test_scaled, y_test)
    weights.append(val_acc)

ensemble = VotingClassifier(
    estimators=[('XGB', xgb_model), ('Poly2', poly_model), ('RFF', rff_model)],
    voting='soft',
    weights=weights
)
ensemble.fit(X_train_scaled, y_train)

# ==============================
# Evaluation Table
# ==============================
models = {
    'XGBoost': xgb_model,
    'Poly2': poly_model,
    'RFF': rff_model,
    'Ensemble': ensemble
}

print("===== Dataset: breast_cancer =====")
for name, model in models.items():
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    f1 = f1_score(y_test, model.predict(X_test_scaled))
    cv_mean = np.mean(cross_val_score(model, X_train_scaled, y_train, cv=5))
    print(f"{name:<8} | Train ACC={train_acc:.4f}, Test ACC={test_acc:.4f}, F1={f1:.4f}, CV mean={cv_mean:.4f}")
