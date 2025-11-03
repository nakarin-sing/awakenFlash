import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin

# -------------------------
# Wrappers
# -------------------------
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

class RFFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=None, n_components=50, random_state=42):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        if self.gamma is None:
            self.gamma = 1.0 / (n_features * np.var(X))
        self.W = rng.normal(0, np.sqrt(2 * self.gamma), size=(n_features, self.n_components))
        self.b = rng.uniform(0, 2*np.pi, size=self.n_components)
        Z = self._transform(X)
        self.model.fit(Z, y)
        return self

    def _transform(self, X):
        return np.sqrt(2/self.n_components) * np.cos(X @ self.W + self.b)

    def predict(self, X):
        return self.model.predict(self._transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._transform(X))

    def score(self, X, y):
        return self.model.score(self._transform(X), y)

# -------------------------
# Load Dataset
# -------------------------
data = load_breast_cancer()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# -------------------------
# Initialize Models
# -------------------------
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
poly_model = Poly2Wrapper(C=1.0)
rff_model = RFFClassifier(n_components=50)

models = {
    'XGBoost': xgb_model,
    'Poly2': poly_model,
    'RFF': rff_model
}

# -------------------------
# Evaluation
# -------------------------
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    cv_mean = cross_val_score(model, X_train, y_train, cv=5).mean()
    
    results.append([name, acc_train, acc_test, f1, cv_mean])

# -------------------------
# Weighted Ensemble (using train CV for weights)
# -------------------------
weights = [cv_mean for _, _, _, _, cv_mean in results]
weights = np.array(weights)/sum(weights)
estimators = [(name, model) for name, model in models.items()]
ensemble = VotingClassifier(estimators=estimators, voting='soft', weights=weights)
ensemble.fit(X_train, y_train)
y_test_pred = ensemble.predict(X_test)
results.append([
    'Ensemble',
    np.nan,
    accuracy_score(y_test, y_test_pred),
    f1_score(y_test, y_test_pred),
    np.nan
])

# -------------------------
# Display Table
# -------------------------
df = pd.DataFrame(results, columns=['Model','Train ACC','Test ACC','F1','CV mean'])
print(df.to_string(index=False))
