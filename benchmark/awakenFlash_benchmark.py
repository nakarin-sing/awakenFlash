import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

# ===== Wrapper Classes =====
class Poly2Wrapper:
    def __init__(self, C=1.0):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.model = LogisticRegression(max_iter=2000, C=C, random_state=42)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

class RFFWrapper:
    def __init__(self, gamma=0.5, n_components=100):
        self.gamma = gamma
        self.n_components = n_components
        self.W = None
        self.model = LogisticRegression(max_iter=2000, random_state=42)

    def fit(self, X, y):
        D = X.shape[1]
        self.W = np.random.normal(0, np.sqrt(2*self.gamma), size=(D, self.n_components))
        Z = np.cos(X @ self.W)
        self.model.fit(Z, y)
        return self

    def predict(self, X):
        Z = np.cos(X @ self.W)
        return self.model.predict(Z)

# ===== Adaptive Hyperparameters =====
def adaptive_hyperparameters(X_train):
    n_samples, n_features = X_train.shape
    gamma = 1.0 / (n_features * np.var(X_train))
    C = 0.1 if n_samples / n_features < 10 else 1.0
    n_components = min(100, n_samples // 2)
    return gamma, C, n_components

# ===== Weighted Ensemble =====
def weighted_ensemble_predict(models, X_val, y_val, X_test):
    weights = []
    for m in models:
        pred_val = m.predict(X_val)
        weights.append(accuracy_score(y_val, pred_val))
    weights = np.array(weights) / sum(weights)
    
    preds = np.array([m.predict(X_test) for m in models])
    final_pred = []
    for i in range(X_test.shape[0]):
        counts = np.bincount(preds[:, i], weights=weights)
        final_pred.append(np.argmax(counts))
    return np.array(final_pred)

# ===== Datasets =====
datasets = {
    "breast_cancer": load_breast_cancer(),
    "iris": load_iris(),
    "wine": load_wine()
}

# ===== Benchmark Loop =====
for name, data in datasets.items():
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # further split for ensemble validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    gamma, C, n_components = adaptive_hyperparameters(X_train_final)

    # Initialize models
    models = {
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Poly2": Poly2Wrapper(C=C),
        "RFF": RFFWrapper(gamma=gamma, n_components=n_components)
    }

    # Train models
    for m in models.values():
        m.fit(X_train_final, y_train_final)

    # Individual results
    print(f"\n===== Dataset: {name} =====")
    for model_name, model in models.items():
        pred_test = model.predict(X_test_scaled)
        pred_train = model.predict(X_train_final)
        acc_test = accuracy_score(y_test, pred_test)
        acc_train = accuracy_score(y_train_final, pred_train)
        f1 = f1_score(y_test, pred_test, average='weighted')
        cv_scores = cross_val_score(model.model if hasattr(model, 'model') else model, X_train_final, y_train_final, cv=5)
        print(f"{model_name:8} | Train ACC={acc_train:.4f}, Test ACC={acc_test:.4f}, F1={f1:.4f}, CV mean={cv_scores.mean():.4f}")

    # Ensemble
    ensemble_pred = weighted_ensemble_predict(list(models.values()), X_val, y_val, X_test_scaled)
    acc_ens = accuracy_score(y_test, ensemble_pred)
    f1_ens = f1_score(y_test, ensemble_pred, average='weighted')
    print(f"{'Ensemble':8} | Test ACC={acc_ens:.4f}, F1={f1_ens:.4f}")
