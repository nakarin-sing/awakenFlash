import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# -------------------------
# Random Fourier Features Wrapper
# -------------------------
class RFFWrapper:
    def __init__(self, gamma=None, n_components=100, random_state=42):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        if self.gamma is None:
            self.gamma = 1.0 / (n_features * np.var(X))
        self.W = rng.normal(0, np.sqrt(2 * self.gamma), size=(n_features, self.n_components))
        self.b = rng.uniform(0, 2 * np.pi, size=self.n_components)
        return self

    def transform(self, X):
        Z = np.sqrt(2 / self.n_components) * np.cos(X @ self.W + self.b)
        return Z

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# -------------------------
# Polynomial Wrapper (degree=2)
# -------------------------
class Poly2Wrapper:
    def __init__(self, C=1.0):
        self.C = C
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.model = LogisticRegression(C=self.C, max_iter=1000)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def predict_proba(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict_proba(X_poly)

# -------------------------
# Load Data
# -------------------------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.63, random_state=42)  # ~211 train, 358 test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Initialize Models
# -------------------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
poly2 = Poly2Wrapper(C=0.1)
rff = RFFWrapper(n_components=50)
X_train_rff = rff.fit_transform(X_train_scaled)
X_test_rff = rff.transform(X_test_scaled)
rff_model = LogisticRegression(max_iter=1000).fit(X_train_rff, y_train)

models = {
    'XGBoost': xgb.fit(X_train_scaled, y_train),
    'Poly2': poly2.fit(X_train_scaled, y_train),
    'RFF': rff_model
}

# -------------------------
# Weighted Ensemble (by validation acc)
# -------------------------
def weighted_ensemble_predict(models, X_val, y_val, X_test):
    weights = []
    for m in models.values():
        pred_val = m.predict(X_val)
        weights.append(accuracy_score(y_val, pred_val))
    weights = np.array(weights) / np.sum(weights)

    # ensemble prediction (weighted sum of probabilities)
    probs = np.zeros((X_test.shape[0], len(np.unique(y_val))))
    for w, m in zip(weights, models.values()):
        if hasattr(m, "predict_proba"):
            probs += w * m.predict_proba(X_test)
        else:  # XGBoost
            probs += w * m.predict_proba(X_test)
    return np.argmax(probs, axis=1)

# -------------------------
# Train/Test Evaluation
# -------------------------
print("===== Dataset: breast_cancer =====")
for name, model in models.items():
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    cv = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"{name:7} | Train ACC={train_acc:.4f}, Test ACC={test_acc:.4f}, F1={f1:.4f}, CV mean={cv.mean():.4f}")

# Ensemble
ensemble_pred = weighted_ensemble_predict(models, X_train_scaled, y_train, X_test_scaled)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred)
print(f"{'Ensemble':7} | Test ACC={ensemble_acc:.4f}, F1={ensemble_f1:.4f}")
