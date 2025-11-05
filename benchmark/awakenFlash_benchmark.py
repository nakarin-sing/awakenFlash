# ==============================================
# Practical Ensemble 2.0 — Speed & Accuracy Optimized
# ==============================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# 1. Load & Prepare Data
# ------------------------
def prepare_data(df, target_col):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # Shuffle for stability
    X, y = shuffle(X, y, random_state=42)
    
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Polynomial Interaction Features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X = poly.fit_transform(X)
    
    return X, y

# ------------------------
# 2. Base Models
# ------------------------
base_models = [
    ('sgd', SGDClassifier(max_iter=1000, tol=1e-3, learning_rate='optimal', n_jobs=-1)),
    ('pa', PassiveAggressiveClassifier(max_iter=50, tol=1e-2)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3))
]

# ------------------------
# 3. Stacking Ensemble
# ------------------------
# Meta model: lightweight, fast
meta_model = SGDClassifier(max_iter=500, tol=1e-3)

practical_ensemble = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    n_jobs=-1,
    passthrough=True
)

# ------------------------
# 4. Training & Benchmark
# ------------------------
def run_benchmark(df, target_col='target'):
    X, y = prepare_data(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    practical_ensemble.fit(X_train, y_train)
    
    # Predict
    y_pred = practical_ensemble.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    
    return acc

# ------------------------
# 5. Example Benchmark Run
# ------------------------
if __name__ == "__main__":
    # Generate synthetic dataset
    from sklearn.datasets import make_classification
    df_list = []
    for i in range(10):  # 10 chunks
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=i)
        df = pd.DataFrame(X, columns=[f'f{j}' for j in range(X.shape[1])])
        df['target'] = y
        df_list.append(df)
    
    results = []
    for idx, df in enumerate(df_list):
        acc = run_benchmark(df)
        print(f"Chunk {idx+1}: Practical Ensemble 2.0 Accuracy = {acc:.3f}")
        results.append(acc)
    
    print("=== Final Average Accuracy ===")
    print(f"Average Accuracy: {np.mean(results):.3f}")
