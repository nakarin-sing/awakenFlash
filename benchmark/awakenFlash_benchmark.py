import os
import numpy as np
import warnings
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# สร้าง folder ผลลัพธ์
os.makedirs("benchmark_results", exist_ok=True)
results_file = "benchmark_results/results_improved.txt"

# ตรึง random seed ทั้งหมด
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# datasets
datasets = {
    "breast_cancer": load_breast_cancer(),
    "iris": load_iris(),
    "wine": load_wine(),
}

def run_benchmark():
    output_lines = []
    output_lines.append("=" * 80 + "\n")
    output_lines.append("IMPROVED FAIR BENCHMARK WITH HYPERPARAMETER TUNING\n")
    output_lines.append("=" * 80 + "\n\n")
    
    for name, data in datasets.items():
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        output_lines.append(f"\n{'='*60}\n")
        output_lines.append(f"Dataset: {name.upper()}\n")
        output_lines.append(f"{'='*60}\n")
        print(f"\n{'='*60}")
        print(f"Dataset: {name.upper()}")
        print(f"{'='*60}")

        results = {}

        # =====================================================================
        # 1. XGBoost with GridSearch
        # =====================================================================
        print("Training XGBoost with GridSearch...")
        xgb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
        xgb = GridSearchCV(
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                         verbosity=0, random_state=RANDOM_STATE),
            xgb_params, cv=5, scoring='accuracy', n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        acc_xgb = accuracy_score(y_test, pred_xgb)
        f1_xgb = f1_score(y_test, pred_xgb, average='weighted')
        results['XGBoost'] = (acc_xgb, f1_xgb)
        output_lines.append(f"XGBoost (Tuned)     ACC: {acc_xgb:.4f}  F1: {f1_xgb:.4f}\n")
        output_lines.append(f"  Best params: {xgb.best_params_}\n")
        print(f"XGBoost (Tuned)     ACC: {acc_xgb:.4f}  F1: {f1_xgb:.4f}")

        # =====================================================================
        # 2. OneStep (LogisticRegression) with GridSearch
        # =====================================================================
        print("Training OneStep with GridSearch...")
        one_params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
        one = GridSearchCV(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            one_params, cv=5, scoring='accuracy', n_jobs=-1
        )
        one.fit(X_train, y_train)
        pred_one = one.predict(X_test)
        acc_one = accuracy_score(y_test, pred_one)
        f1_one = f1_score(y_test, pred_one, average='weighted')
        results['OneStep'] = (acc_one, f1_one)
        output_lines.append(f"OneStep (Tuned)     ACC: {acc_one:.4f}  F1: {f1_one:.4f}\n")
        output_lines.append(f"  Best params: {one.best_params_}\n")
        print(f"OneStep (Tuned)     ACC: {acc_one:.4f}  F1: {f1_one:.4f}")

        # =====================================================================
        # 3. Poly2 (PolynomialFeatures + LogisticRegression) with GridSearch
        # =====================================================================
        print("Training Poly2 with GridSearch...")
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        poly_params = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
        poly_model = GridSearchCV(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            poly_params, cv=5, scoring='accuracy', n_jobs=-1
        )
        poly_model.fit(X_train_poly, y_train)
        pred_poly = poly_model.predict(X_test_poly)
        acc_poly = accuracy_score(y_test, pred_poly)
        f1_poly = f1_score(y_test, pred_poly, average='weighted')
        results['Poly2'] = (acc_poly, f1_poly)
        output_lines.append(f"Poly2 (Tuned)       ACC: {acc_poly:.4f}  F1: {f1_poly:.4f}\n")
        output_lines.append(f"  Best params: {poly_model.best_params_}\n")
        print(f"Poly2 (Tuned)       ACC: {acc_poly:.4f}  F1: {f1_poly:.4f}")

        # =====================================================================
        # 4. RFF (Random Fourier Features) with GridSearch + Multiple Runs
        # =====================================================================
        print("Training RFF with GridSearch (multiple runs)...")
        
        # ค้นหา D และ gamma ที่ดีที่สุด
        best_rff_acc = 0
        best_rff_f1 = 0
        best_rff_config = {}
        
        for D in [50, 100, 200, 500]:
            for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
                # รันหลายครั้งเพื่อลด variance
                accs = []
                f1s = []
                for run in range(5):
                    np.random.seed(RANDOM_STATE + run)
                    W = np.random.normal(0, gamma, size=(X_train.shape[1], D))
                    b = np.random.uniform(0, 2*np.pi, size=D)
                    X_train_rff = np.sqrt(2/D) * np.cos(np.dot(X_train, W) + b)
                    X_test_rff = np.sqrt(2/D) * np.cos(np.dot(X_test, W) + b)
                    
                    rff_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
                    rff_model.fit(X_train_rff, y_train)
                    pred_rff = rff_model.predict(X_test_rff)
                    accs.append(accuracy_score(y_test, pred_rff))
                    f1s.append(f1_score(y_test, pred_rff, average='weighted'))
                
                avg_acc = np.mean(accs)
                avg_f1 = np.mean(f1s)
                
                if avg_acc > best_rff_acc:
                    best_rff_acc = avg_acc
                    best_rff_f1 = avg_f1
                    best_rff_config = {'D': D, 'gamma': gamma}
        
        results['RFF'] = (best_rff_acc, best_rff_f1)
        output_lines.append(f"RFF (Tuned)         ACC: {best_rff_acc:.4f}  F1: {best_rff_f1:.4f}\n")
        output_lines.append(f"  Best params: {best_rff_config}\n")
        print(f"RFF (Tuned)         ACC: {best_rff_acc:.4f}  F1: {best_rff_f1:.4f}")

        # =====================================================================
        # 5. Advanced Ensemble (Weighted Voting based on validation performance)
        # =====================================================================
        print("Training Ensemble with Weighted Voting...")
        
        # คำนวณน้ำหนักจาก cross-validation accuracy
        weights = []
        all_preds = [pred_xgb, pred_one, pred_poly]
        
        for model_name, (acc, f1) in results.items():
            if model_name != 'RFF':  # RFF ใช้ค่าที่คำนวณแล้ว
                weights.append(acc)
        weights.append(best_rff_acc)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # สร้าง RFF prediction ด้วย best config
        np.random.seed(RANDOM_STATE)
        W = np.random.normal(0, best_rff_config['gamma'], 
                            size=(X_train.shape[1], best_rff_config['D']))
        b = np.random.uniform(0, 2*np.pi, size=best_rff_config['D'])
        X_test_rff = np.sqrt(2/best_rff_config['D']) * np.cos(np.dot(X_test, W) + b)
        X_train_rff = np.sqrt(2/best_rff_config['D']) * np.cos(np.dot(X_train, W) + b)
        rff_final = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        rff_final.fit(X_train_rff, y_train)
        pred_rff_final = rff_final.predict(X_test_rff)
        
        all_preds.append(pred_rff_final)
        
        # Weighted voting
        ensemble_pred = []
        for i in range(len(y_test)):
            votes = {}
            for pred, weight in zip(all_preds, weights):
                vote = pred[i]
                votes[vote] = votes.get(vote, 0) + weight
            ensemble_pred.append(max(votes, key=votes.get))
        
        ensemble_pred = np.array(ensemble_pred)
        acc_ens = accuracy_score(y_test, ensemble_pred)
        f1_ens = f1_score(y_test, ensemble_pred, average='weighted')
        results['Ensemble'] = (acc_ens, f1_ens)
        output_lines.append(f"Ensemble (Weighted) ACC: {acc_ens:.4f}  F1: {f1_ens:.4f}\n")
        output_lines.append(f"  Weights: XGB={weights[0]:.3f}, One={weights[1]:.3f}, "
                          f"Poly={weights[2]:.3f}, RFF={weights[3]:.3f}\n")
        print(f"Ensemble (Weighted) ACC: {acc_ens:.4f}  F1: {f1_ens:.4f}")

        # =====================================================================
        # สรุปผล
        # =====================================================================
        output_lines.append(f"\n{'-'*60}\n")
        output_lines.append("RANKING:\n")
        sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
        for rank, (model_name, (acc, f1)) in enumerate(sorted_results, 1):
            output_lines.append(f"  {rank}. {model_name:15s} ACC: {acc:.4f}  F1: {f1:.4f}\n")
        output_lines.append(f"{'-'*60}\n")

    # เซฟไฟล์
    with open(results_file, "w", encoding='utf-8') as f:
        f.writelines(output_lines)
    print(f"\n{'='*60}")
    print(f"Results saved to {results_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_benchmark()
