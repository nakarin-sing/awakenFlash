# awakenFlash_benchmark_refactored.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# ==========================
# Custom Wrappers
# ==========================
class Poly2Wrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, degree=2, C=0.1, interaction_only=True):
        self.degree = degree
        self.C = C
        self.interaction_only = interaction_only

    def fit(self, X, y):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LogisticRegression

        self.poly = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        self.model = LogisticRegression(C=self.C, max_iter=1000, solver='liblinear')
        self.model.fit(X_poly, y)
        return self

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def predict_proba(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict_proba(X_poly)

class RFFWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, gamma='scale', n_components=100, C=1.0):
        self.gamma = gamma
        self.n_components = n_components
        self.C = C

    def fit(self, X, y):
        from sklearn.kernel_approximation import RBFSampler
        from sklearn.linear_model import LogisticRegression

        self.rbf = RBFSampler(gamma=self.gamma, n_components=self.n_components, random_state=42)
        X_rff = self.rbf.fit_transform(X)
        self.model = LogisticRegression(C=self.C, max_iter=1000, solver='liblinear')
        self.model.fit(X_rff, y)
        return self

    def predict(self, X):
        X_rff = self.rbf.transform(X)
        return self.model.predict(X_rff)

    def predict_proba(self, X):
        X_rff = self.rbf.transform(X)
        return self.model.predict_proba(X_rff)

# ==========================
# Load Data
# ==========================
data = load_breast_cancer()
X, y = data.data, data.target

# Standard train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================
# Define Models
# ==========================
models = {
    "XGBoost": xgb.XGBClassifier(
        max_depth=3, n_estimators=200, use_label_encoder=False,
        eval_metric='logloss', random_state=42
    ),
    "Poly2": Poly2Wrapper(degree=2, C=0.1, interaction_only=True),
    "RFF": RFFWrapper(gamma='scale', n_components=100, C=1.0)
}

# ==========================
# Benchmark & Cross-Validation
# ==========================
results = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # pipeline ensures scaling inside CV
    pipeline = make_pipeline(StandardScaler(), model)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    results.append({
        "Model": name,
        "Train ACC": accuracy_score(y_train, pipeline.predict(X_train)),
        "Test ACC": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average='weighted'),
        "CV mean": cv_scores.mean()
    })

# ==========================
# Ensemble
# ==========================
ensemble = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='hard'
)
ensemble_pipeline = make_pipeline(StandardScaler(), ensemble)
ensemble_pipeline.fit(X_train, y_train)
y_pred_ens = ensemble_pipeline.predict(X_test)

results.append({
    "Model": "Ensemble",
    "Train ACC": accuracy_score(y_train, ensemble_pipeline.predict(X_train)),
    "Test ACC": accuracy_score(y_test, y_pred_ens),
    "F1": f1_score(y_test, y_pred_ens, average='weighted'),
    "CV mean": np.nan  # Ensemble CV skipped for simplicity
})

# ==========================
# Show Results
# ==========================
df_results = pd.DataFrame(results)
print(df_results)
