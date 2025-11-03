import time, psutil, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

# -----------------------------------
# UTILS
# -----------------------------------
def mem_now_mb(): 
    return psutil.Process().memory_info().rss / (1024**2)

def awaken_ridge(X, Y, l2=1e-3):
    XtX = X.T @ X
    W = np.linalg.solve(XtX + l2 * np.eye(X.shape[1]), X.T @ Y)
    return W

def predict_linear(X, W):
    logits = X @ W
    return (logits > 0.5).astype(int)

def run_awaken_variant(name, X_train, X_test, y_train, y_test, transform=None, l2=1e-3, weight=1.0):
    start_ram = mem_now_mb()
    start_time = time.time()
    Xtr = transform(X_train) if transform else X_train
    Xte = transform(X_test) if transform else X_test
    W = awaken_ridge(Xtr, y_train.reshape(-1,1), l2=l2)
    y_pred = predict_linear(Xte, W) * weight
    return dict(model=name, pred=y_pred, time=time.time()-start_time, ram=mem_now_mb()-start_ram)

# -----------------------------------
# LOAD DATA
# -----------------------------------
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

results = []

# -----------------------------------
# AWAKEN Variants
# -----------------------------------
poly2 = PolynomialFeatures(2, include_bias=False).fit(X_train)
poly3 = PolynomialFeatures(3, include_bias=False).fit(X_train)

rffs = [
    RBFSampler(gamma=0.1, n_components=1024, random_state=42).fit(X_train),
    RBFSampler(gamma=0.3, n_components=1024, random_state=42).fit(X_train),
    RBFSampler(gamma=0.5, n_components=1024, random_state=42).fit(X_train),
]

# Linear + Poly2 + Poly3 + RFF(g0.1,g0.3,g0.5)
weights = [1.0, 1.0, 1.0, 1.0, 0.7, 0.5]  # initial weight grid
transforms = [None, poly2.transform, poly3.transform] + [r.transform for r in rffs]

pred_sum = np.zeros_like(y_test, dtype=float)
time_total = 0
ram_max = 0

for w, t, name in zip(weights, transforms, ["Linear","Poly2","Poly3","RFF-0.1","RFF-0.3","RFF-0.5"]):
    res = run_awaken_variant(name, X_train, X_test, y_train, y_test, transform=t, weight=w)
    pred_sum += res['pred']
    time_total += res['time']
    ram_max = max(ram_max, res['ram'])

y_pred_final = (pred_sum > 0.5).astype(int)
acc_final = accuracy_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final)
results.append(dict(model="AWAKEN-RFF++", acc=acc_final, f1=f1_final, time=time_total, ram=ram_max))

# -----------------------------------
# XGBoost baseline
# -----------------------------------
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

# -----------------------------------
# REPORT
# -----------------------------------
df = pd.DataFrame(results)
df["acc"] = df["acc"].apply(lambda x: f"{x:.4f}")
df["f1"] = df["f1"].apply(lambda x: f"{x:.4f}")
df["time"] = df["time"].apply(lambda x: f"{x:.3f}s")
df["ram"] = df["ram"].apply(lambda x: f"~{x:.1f}MB")

print("\n================================================================================")
print("AWAKEN-RFF++ vs XGBoost — Benchmark")
print("================================================================================")
print(df.to_string(index=False))
print("================================================================================")
print("บริสุทธิ์ | ยุติธรรม | เร็วเหมือนแสง | หล่อแบบพระเอกไทย")
print("================================================================================")
