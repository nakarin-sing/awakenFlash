import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
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
    def __init__(self, C=1.0):
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
    # Random Fourier Features (simplified)
    def __init__(self, gamma=0.5, n_components=500):
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
# Weighted Ensemble
# ===============================
def weighted_ensemble_predict(probas_list, weights=None):
    probas_array = np.array(probas_list)  # shape: [n_models, n_samples, n_classes]
    if weights is None:
        weights = np.ones(probas_array.shape[0])
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

    results = {}

    for name, dataset in datasets.items():
        X, y = dataset.data, dataset.target

        # Split & scale properly (no leakage)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize models
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        ones = OneStepWrapper()
        poly = Poly2Wrapper(C=0.5)
        rff = RFFWrapper(gamma=0.5, n_components=500)

        models = [("XGBoost", xgb), ("OneStep", ones), ("Poly2", poly), ("RFF", rff)]

        # Fit models
        for _, model in models:
            model.fit(X_train_scaled, y_train)

        # Predict & metrics
        model_preds = {}
        model_probas = []
        for name_m, model in models:
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)
            model_preds[name_m] = {
                "ACC": accuracy_score(y_test, pred),
                "F1": f1_score(y_test, pred, average='weighted'),
                "Report": classification_report(y_test, pred, zero_division=0)
            }
            model_probas.append(prob)

        # Weighted ensemble
        ensemble_pred = weighted_ensemble_predict(model_probas)
        model_preds["Ensemble"] = {
            "ACC": accuracy_score(y_test, ensemble_pred),
            "F1": f1_score(y_test, ensemble_pred, average='weighted'),
            "Report": classification_report(y_test, ensemble_pred, zero_division=0)
        }

        results[name] = model_preds

    # Save results
    import os
    os.makedirs("benchmark_results", exist_ok=True)
    with open("benchmark_results/results.txt", "w") as f:
        for dname, res in results.items():
            f.write(f"===== Dataset: {dname} =====\n")
            for mname, metrics in res.items():
                f.write(f"{mname}: ACC={metrics['ACC']:.4f}, F1={metrics['F1']:.4f}\n")
            f.write("\n")
    print("Results saved to benchmark_results/results.txt")
    return results

# ===============================
# Run benchmark
# ===============================
if __name__ == "__main__":
    run_benchmark()
