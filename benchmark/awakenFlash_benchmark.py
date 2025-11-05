import pandas as pd
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ===============================
# เตรียม dataset ตัวอย่าง 10 chunks
# ===============================
df_list = []
for seed in range(10):
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=seed
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df['target'] = y
    df_list.append(df)

# ===============================
# ฟังก์ชัน Practical Ensemble 2.0
# ===============================
def practical_ensemble_train(X_train, y_train):
    # Ensemble: SGD + Perceptron
    sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    pa = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('sgd', sgd), ('pa', pa)],
        voting='hard'
    )
    ensemble.fit(X_train, y_train)
    return ensemble

# ===============================
# Benchmark loop
# ===============================
practical_accuracies = []
xgb_accuracies = []
practical_times = []
xgb_times = []

for idx, df in enumerate(df_list):
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Practical Ensemble
    start = time.time()
    model_practical = practical_ensemble_train(X_train, y_train)
    y_pred = model_practical.predict(X_test)
    practical_acc = accuracy_score(y_test, y_pred)
    practical_time = time.time() - start

    practical_accuracies.append(practical_acc)
    practical_times.append(practical_time)

    # XGBoost
    start = time.time()
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model_xgb.fit(X_train, y_train)
    y_pred_xgb = model_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_time = time.time() - start

    xgb_accuracies.append(xgb_acc)
    xgb_times.append(xgb_time)

    # Print per chunk
    winner = "Practical Ensemble" if practical_acc >= xgb_acc else "XGBoost"
    print(f"Chunk {idx+1}: Practical={practical_acc:.3f} | XGBoost={xgb_acc:.3f} | 🏆 {winner}")

# ===============================
# Final Results
# ===============================
print("\n=== Final Average Results ===")
print(f"Average Practical Accuracy: {np.mean(practical_accuracies):.3f}")
print(f"Average XGBoost Accuracy:   {np.mean(xgb_accuracies):.3f}")
print(f"Average Practical Time:     {np.mean(practical_times):.3f}s")
print(f"Average XGBoost Time:       {np.mean(xgb_times):.3f}s")
