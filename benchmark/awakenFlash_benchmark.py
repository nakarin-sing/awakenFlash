import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

# ===============================
# Wrapper Classes
# ===============================
class OneStepWrapper:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class Poly2Wrapper:
    def __init__(self, C=0.5):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.model = LogisticRegression(max_iter=1000, C=C, random_state=42)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)

    def predict(self, X):
        return self.model.predict(self.poly.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.poly.transform(X))

class RFFWrapper:
    def __init__(self, gamma=0.5, n_components=100):
        self.gamma = gamma
        self.n_components = n_components
        self.model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def fit(self, X, y):
        np.random.seed(42)
        D = X.shape[1]
        self.W = np.random.normal(0, np.sqrt(2*self.gamma), size=(D, self.n_components))
        self.b = np.random.uniform(0, 2*np.pi, size=self.n_components)
        X_rff = np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)
        self.model.fit(X_rff, y)

    def transform(self, X):
        return np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)

    def predict(self, X):
        return self.model.predict(self.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.transform(X))

# ===============================
# Weighted Ensemble (by validation accuracy)
# ===============================
def weighted_ensemble_predict(models, X_val, y_val, X_test):
    # Compute weights from validation accuracy
    weights = []
    probas_list = []
    for model in models:
        pred_val = model.predict(X_val)
        acc_val = accuracy_score(y_val, pred_val)
        weights.append(acc_val)
        probas_list.append(model.predict_proba(X_test))
    weights = np.array(weights) / sum(weights)
    probas_array = np.array(probas_list)
    weighted_avg = np.tensordot(weights, probas_array, axes=(0,0))
    return np.argmax(weighted_avg, axis=1)

# ===============================
# Benchmark Function
# ===============================
def run_benchmark():
    datasets = {
        "breast_cancer": load_breast_cancer(),
        "iris": load_iris(),
        "wine": load_wine()
    }

    for name, dataset in datasets.items():
        X, y = dataset.data, dataset.target
        # Split train/test (70/30)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Split a small validation set from train for ensemble weights
        X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train)

        # Initialize models
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        ones = OneStepWrapper()
        poly = Poly2Wrapper(C=0.5)
        rff = RFFWrapper(gamma=0.5, n_components=100)

        models = [("XGBoost", xgb), ("OneStep", ones), ("Poly2", poly), ("RFF", rff)]

        # Fit models
        for _, model in models:
            model.fit(X_train_final, y_train_final)

        print(f"\n===== Dataset: {name} =====")
        for mname, model in models:
            pred = model.predict(X_test_scaled)
            train_pred = model.predict(X_train_final)
            acc_train = accuracy_score(y_train_final, train_pred)
            acc_test = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='weighted')
            print(f"{mname}: Train ACC={acc_train:.4f}, Test ACC={acc_test:.4f}, F1={f1:.4f}")

        # Ensemble
        ensemble_pred = weighted_ensemble_predict([xgb, ones, poly, rff], X_val, y_val, X_test_scaled)
        acc_ens = accuracy_score(y_test, ensemble_pred)
        f1_ens = f1_score(y_test, ensemble_pred, average='weighted')
        print(f"Ensemble (weighted): Test ACC={acc_ens:.4f}, F1={f1_ens:.4f}")

# ===============================
# Run benchmark
# ===============================
if __name__ == "__main__":
    run_benchmark()
