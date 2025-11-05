import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# สมมติ run_benchmark() คือฟังก์ชัน Practical Ensemble 2.0 ของคุณ
def run_practical(df, target_col='target'):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    start = time.time()
    practical_acc = run_benchmark(df)  # เรียก Practical Ensemble 2.0 ของคุณ
    duration = time.time() - start
    
    return practical_acc, duration

def run_xgboost(df, target_col='target'):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    start = time.time()
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    duration = time.time() - start
    
    return acc, duration

# ------------------------
# Benchmark comparison
# ------------------------
practical_results, practical_times = [], []
xgb_results, xgb_times = [], []

for idx, df in enumerate(df_list):
    p_acc, p_time = run_practical(df)
    x_acc, x_time = run_xgboost(df)
    
    practical_results.append(p_acc)
    practical_times.append(p_time)
    xgb_results.append(x_acc)
    xgb_times.append(x_time)
    
    winner = "🏆 Practical" if p_acc > x_acc else "🎯 XGBoost" if x_acc > p_acc else "⚖️ Tie"
    print(f"Chunk {idx+1}: Practical = {p_acc:.3f} ({p_time:.2f}s) | "
          f"XGBoost = {x_acc:.3f} ({x_time:.2f}s) | {winner}")

print("\n=== Final Averages ===")
print(f"Practical Ensemble 2.0: Accuracy = {np.mean(practical_results):.3f}, Time = {np.mean(practical_times):.2f}s")
print(f"XGBoost: Accuracy = {np.mean(xgb_results):.3f}, Time = {np.mean(xgb_times):.2f}s")
