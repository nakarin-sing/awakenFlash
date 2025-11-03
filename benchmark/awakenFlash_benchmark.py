import os
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# =========================
# Model Wrappers
# =========================

class XGBoostWrapper:
    def __init__(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class OneStepWrapper:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)

class Poly2Wrapper:
    def __init__(self):
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

class RFFWrapper:
    def __init__(self, gamma=1.0, n_components=100):
        self.rff = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    def fit(self, X, y):
        X_rff = self.rff.fit_transform(X)
        self.model.fit(X_rff, y)
    def predict(self, X):
        X_rff = self.rff.transform(X)
        return self.model.predict(X_rff)

# =========================
# Benchmark Runner
# =========================

def run_benchmark():
    datasets = {
        "breast_cancer": load_breast_cancer(),
        "iris": load_iris(),
        "wine": load_wine()
    }

    os.makedirs("benchmark_results", exist_ok=True)
    results_file = "benchmark_results/results.txt"

    with open(results_file, "w") as f:
        for name, data in datasets.items():
            X, y = data.data, data.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler().fit(X_train)
            X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

            # Initialize models
            xgb_model = XGBoostWrapper(); xgb_model.fit(X_train, y_train); xgb_pred = xgb_model.predict(X_test)
            one = OneStepWrapper(); one.fit(X_train, y_train); one_pred = one.predict(X_test)
            poly = Poly2Wrapper(); poly.fit(X_train, y_train); poly_pred = poly.predict(X_test)
            rff = RFFWrapper(); rff.fit(X_train, y_train); rff_pred = rff.predict(X_test)

            # Ensemble: majority voting
            ensemble_preds = np.array([xgb_pred, one_pred, poly_pred, rff_pred])
            ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, ensemble_preds)

            # Evaluate all models
            models = {
                "XGBoost": xgb_pred,
                "OneStep": one_pred,
                "Poly2": poly_pred,
                "RFF": rff_pred,
                "Ensemble": ensemble_pred
            }

            log = [f"===== Dataset: {name} ====="]
            for mname, pred in models.items():
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average='weighted')
                log.append(f"{mname:10} ACC: {acc:.4f} F1: {f1:.4f}")
            log_text = "\n".join(log)
            print(log_text)
            f.write(log_text + "\n\n")

    print("="*70)
    print("AWAKEN vΩ.Real+++ — Real XGBoost + Logistic + Poly + RFF Ensemble | CI-ready")
    print(f"Results saved to {results_file}")
    print("="*70)

if __name__ == "__main__":
    run_benchmark()
