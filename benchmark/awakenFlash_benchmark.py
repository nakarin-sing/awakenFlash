# awakenFlash_benchmark_real.py — Real-World Ensemble Benchmark
import os
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.kernel_approximation import RBFSampler

# =======================
# Random Fourier Features wrapper
# =======================
class RFF:
    def __init__(self, n_components=100, gamma=1.0):
        self.n_components = n_components
        self.gamma = gamma
        self.rbf_sampler = RBFSampler(n_components=self.n_components, gamma=self.gamma, random_state=42)
        self.model = LogisticRegression(max_iter=500)

    def fit(self, X, y):
        X_new = self.rbf_sampler.fit_transform(X)
        self.model.fit(X_new, y)

    def predict(self, X):
        X_new = self.rbf_sampler.transform(X)
        return self.model.predict(X_new)

# =======================
# Benchmark function
# =======================
def run_benchmark():
    datasets = {
        "breast_cancer": load_breast_cancer(),
        "iris": load_iris(),
        "wine": load_wine()
    }

    os.makedirs("benchmark_results", exist_ok=True)
    results = []

    for name, dataset in datasets.items():
        X, y = dataset.data, dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Models
        models = {}

        # XGBoost
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        # Logistic Regression (OneStep)
        models['OneStep'] = LogisticRegression(max_iter=500, random_state=42)
        # Polynomial Features + Logistic Regression (Poly2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        poly_model = LogisticRegression(max_iter=500, random_state=42)
        models['Poly2'] = (poly_model, X_train_poly, X_test_poly)
        # RFF
        models['RFF'] = RFF(n_components=100, gamma=1.0)

        preds = {}
        log_lines = [f"===== Dataset: {name} ====="]

        # Train and predict
        for m_name, model in models.items():
            if m_name == 'Poly2':
                clf, X_tr, X_te = model
                clf.fit(X_tr, y_train)
                y_pred = clf.predict(X_te)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            preds[m_name] = y_pred
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            log_lines.append(f"{m_name:<10} ACC: {acc:.4f}  F1: {f1:.4f}")
            results.append(f"{name},{m_name},{acc:.4f},{f1:.4f}")

        # Ensemble: majority vote of OneStep + Poly2 + RFF
        ensemble_preds = np.array([preds['OneStep'], preds['Poly2'], preds['RFF']])
        ensemble_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=ensemble_preds)
        acc_ens = accuracy_score(y_test, ensemble_vote)
        f1_ens = f1_score(y_test, ensemble_vote, average='weighted')
        log_lines.append(f"{'Ensemble':<10} ACC: {acc_ens:.4f}  F1: {f1_ens:.4f}")
        results.append(f"{name},Ensemble,{acc_ens:.4f},{f1_ens:.4f}")

        # Print log to CI
        print("\n".join(log_lines))
        print("="*70)

    # Save results to file
    with open("benchmark_results/results.txt", "w") as f:
        f.write("\n".join(results))
    print("Results saved to benchmark_results/results.txt")

# =======================
if __name__ == "__main__":
    run_benchmark()
