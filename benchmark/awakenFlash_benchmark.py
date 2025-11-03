# awaken_realworld_benchmark.py
# =============================================
# FAIR BENCHMARK — AWAKEN vΩ vs XGBoost (UCI)
# =============================================

import time, psutil, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score
from sklearn.kernel_approximation import RBFSampler
import xgboost as xgb

def mem_now_mb():
    return psutil.Process().memory_info().rss / (1024 ** 2)

def awaken_pinv(X, Y):
    W = np.linalg.pinv(X) @ Y
    return W

def awaken_ridge(X, Y, l2=1e-2):
    XtX = X.T @ X
    W = np.linalg.solve(XtX + l2 * np.eye(X.shape[1]), X.T @ Y)
    return W

def predict_linear(X, W):
    return (X @ W > 0.5).astype(int)

def run_awaken_variant(name, X_train, X_test, y_train, y_test, transform=None):
    start_ram = mem_now_mb()
    start_time = time.time()

    Xtr = transform(X_train) if transform else X_train
    Xte = transform(X_test) if transform else X_test

    W = awaken_ridge(Xtr, y_train.reshape(-1,1), l2=1e-2)
    y_pred = predict_linear(Xte, W)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    end_time = time.time()
    end_ram = mem_now_mb()
    return dict(model=name, acc=acc, f1=f1, time=end_time-start_time, ram=end_ram-start_ram)

# ===================== LOAD DATA =====================
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# ===================== RUN MODELS =====================
results = []

# 1️⃣ AWAKEN Pure (pinv)
results.append(run_awaken_variant("AWAKEN-Pure (pinv)", X_train, X_test, y_train, y_test))

# 2️⃣ AWAKEN Ridge
results.append(run_awaken_variant("AWAKEN-Ridge (λ=1e-2)", X_train, X_test, y_train, y_test))

# 3️⃣ AWAKEN + Poly(2)
poly = PolynomialFeatures(2, include_bias=False).fit(X_train)
results.append(run_awaken_variant("AWAKEN-Poly2", X_train, X_test, y_train, y_test, transform=poly.transform))

# 4️⃣ AWAKEN + RFF (approx RBF)
rff = RBFSampler(gamma=0.3, n_components=512, random_state=0).fit(X_train)
results.append(run_awaken_variant("AWAKEN-RFF512", X_train, X_test, y_train, y_test, transform=rff.transform))

# 5️⃣ XGBoost baseline
start_ram = mem_now_mb()
start_time = time.time()
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results.append(dict(model="XGBoost", acc=acc, f1=f1,
                    time=time.time()-start_time, ram=mem_now_mb()-start_ram))

# ===================== REPORT =====================
df = pd.DataFrame(results)
df["acc"] = df["acc"].apply(lambda x: f"{x:.4f}")
df["f1"] = df["f1"].apply(lambda x: f"{x:.4f}")
df["time"] = df["time"].apply(lambda x: f"{x:.3f}s")
df["ram"] = df["ram"].apply(lambda x: f"~{x:.1f}MB")

print("\n================================================================================")
print("AWAKEN vΩ — REAL WORLD BENCHMARK (Breast Cancer)")
print("================================================================================")
print(df.to_string(index=False))
print("================================================================================")
print("บริสุทธิ์ | ยุติธรรม | เร็วเหมือนแสง | หล่อแบบพระเอกไทย")
print("================================================================================")
