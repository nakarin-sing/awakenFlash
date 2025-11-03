import os
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# สร้าง folder ผลลัพธ์
os.makedirs("benchmark_results", exist_ok=True)
results_file = "benchmark_results/results.txt"

# datasets
datasets = {
    "breast_cancer": load_breast_cancer(),
    "iris": load_iris(),
    "wine": load_wine(),
}

def run_benchmark():
    output_lines = []
    for name, data in datasets.items():
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        output_lines.append(f"\n===== Dataset: {name} =====\n")
        print(f"\n===== Dataset: {name} =====\n")

        # โมเดล XGBoost
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        f1_xgb = f1_score(y_test, pred_xgb, average='weighted')
        output_lines.append(f"XGBoost    ACC: {acc_xgb:.4f}  F1: {f1_xgb:.4f}\n")
        print(f"XGBoost    ACC: {acc_xgb:.4f}  F1: {f1_xgb:.4f}")

        # โมเดล OneStep (LogisticRegression)
        one = LogisticRegression(max_iter=1000)
        one.fit(X_train, y_train)
        pred_one = one.predict(X_test)
        acc_one = accuracy_score(y_test, pred_one)
        f1_one = f1_score(y_test, pred_one, average='weighted')
        output_lines.append(f"OneStep    ACC: {acc_one:.4f}  F1: {f1_one:.4f}\n")
        print(f"OneStep    ACC: {acc_one:.4f}  F1: {f1_one:.4f}")

        # Poly2 (PolynomialFeatures + LogisticRegression)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        poly_model = LogisticRegression(max_iter=1000)
        poly_model.fit(X_train_poly, y_train)
        pred_poly = poly_model.predict(X_test_poly)
        acc_poly = accuracy_score(y_test, pred_poly)
        f1_poly = f1_score(y_test, pred_poly, average='weighted')
        output_lines.append(f"Poly2      ACC: {acc_poly:.4f}  F1: {f1_poly:.4f}\n")
        print(f"Poly2      ACC: {acc_poly:.4f}  F1: {f1_poly:.4f}")

        # RFF (Random Fourier Features + LogisticRegression)
        D = 100  # จำนวน features ใหม่
        W = np.random.normal(size=(X_train.shape[1], D))
        b = np.random.uniform(0, 2*np.pi, size=D)
        X_train_rff = np.sqrt(2/D) * np.cos(np.dot(X_train, W) + b)
        X_test_rff = np.sqrt(2/D) * np.cos(np.dot(X_test, W) + b)
        rff_model = LogisticRegression(max_iter=1000)
        rff_model.fit(X_train_rff, y_train)
        pred_rff = rff_model.predict(X_test_rff)
        acc_rff = accuracy_score(y_test, pred_rff)
        f1_rff = f1_score(y_test, pred_rff, average='weighted')
        output_lines.append(f"RFF        ACC: {acc_rff:.4f}  F1: {f1_rff:.4f}\n")
        print(f"RFF        ACC: {acc_rff:.4f}  F1: {f1_rff:.4f}")

        # Ensemble (majority vote)
        ensemble_preds = np.array([pred_xgb, pred_one, pred_poly, pred_rff])
        ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=ensemble_preds)
        acc_ens = accuracy_score(y_test, ensemble_pred)
        f1_ens = f1_score(y_test, ensemble_pred, average='weighted')
        output_lines.append(f"Ensemble   ACC: {acc_ens:.4f}  F1: {f1_ens:.4f}\n")
        print(f"Ensemble   ACC: {acc_ens:.4f}  F1: {f1_ens:.4f}")

    # เซฟไฟล์
    with open(results_file, "w") as f:
        f.writelines(output_lines)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    run_benchmark()
