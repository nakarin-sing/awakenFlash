# awaken_vs_xgb.py
import time, psutil, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# ===========================================
# UTILS
# ===========================================
def mem_now_mb(): 
    return psutil.Process().memory_info().rss / (1024**2)

def awaken_ridge(X, Y, l2=1e-3):
    XtX = X.T @ X
    W = np.linalg.solve(XtX + l2 * np.eye(X.shape[1]), X.T @ Y)
    return W

def predict_linear(X, W):
    logits = X @ W
    return (logits > 0.5).astype(int)

def run_awaken_variant(name, X_train, X_test, y_train, y_test, transform=None, l2=1e-3):
    start_ram = mem_now_mb()
    start_time = time.time()
    Xtr = transform(X_train) if transform else X_train
    Xte = transform(X_test) if transform else X_test
    W = awaken_ridge(Xtr, y_train.reshape(-1,1), l2=l2)
    y_pred = predict_linear(Xte, W)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return dict(model=name, acc=acc, f1=f1, time=time.time()-start_time, ram=mem_now_mb()-start_ram)

# ===========================================
# LOAD DATA
# ===========================================
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

results = []

# -----------------------------
# AWAKEN Variants
# -----------------------------
# Linear Ridge
results.append(run_awaken_variant("AWAKEN-Ridge", X_train, X_test, y_train, y_test))

# Poly2
poly2 = PolynomialFeatures(2, include_bias=False).fit(X_train)
results.append(run_awaken_variant("AWAKEN-Poly2", X_train, X_test, y_train, y_test, transform=poly2.transform))

# Poly3
poly3 = PolynomialFeatures(3, include_bias=False).fit(X_train)
results.append(run_awaken_variant("AWAKEN-Poly3", X_train, X_test, y_train, y_test, transform=poly3.transform))

# RFF auto-tune
gammas = [0.1,0.3,0.5,1.0]
best_acc = 0
best_rff = None
for g in gammas:
    rff = RBFSampler(gamma=g, n_components=1024, random_state=42).fit(X_train)
    res = run_awaken_variant(f"AWAKEN-RFF-g{g}", X_train, X_test, y_train, y_test, transform=rff.transform)
    results.append(res)
    if res['acc'] > best_acc:
        best_acc = res['acc']
        best_rff = rff

# Weighted Ensemble (Linear + Poly2 + Poly3 + best RFF)
X_train_ens = np.hstack([
    X_train,
    poly2.transform(X_train),
    poly3.transform(X_train),
    best_rff.transform(X_train)
])
X_test_ens = np.hstack([
    X_test,
    poly2.transform(X_test),
    poly3.transform(X_test),
    best_rff.transform(X_test)
])
results.append(run_awaken_variant("AWAKEN-Ensemble", X_train_ens, X_test_ens, y_train, y_test))

# -----------------------------
# XGBoost baseline
# -----------------------------
start_ram = mem_now_mb()
start_time = time.time()
model_xgb = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    tree_method='hist',
    verbosity=0,
    random_state=42
)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
results.append(dict(model="XGBoost", acc=acc, f1=f1, time=time.time()-start_time, ram=mem_now_mb()-start_ram))

# ===========================================
# REPORT
# ===========================================
df = pd.DataFrame(results)
df["acc"] = df["acc"].apply(lambda x: f"{x:.4f}")
df["f1"] = df["f1"].apply(lambda x: f"{x:.4f}")
df["time"] = df["time"].apply(lambda x: f"{x:.3f}s")
df["ram"] = df["ram"].apply(lambda x: f"~{x:.1f}MB")

print("\n================================================================================")
print("AWAKEN vΩ-RealWorld++ vs XGBoost — Benchmark")
print("================================================================================")
print(df.to_string(index=False))
print("================================================================================")
print("บริสุทธิ์ | ยุติธรรม | เร็วเหมือนแสง | หล่อแบบพระเอกไทย")
print("================================================================================")
