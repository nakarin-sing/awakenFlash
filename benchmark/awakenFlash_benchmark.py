import numpy as np
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from awakenFlash import AdaptiveSRLS
from awakenFlash.datasets import load_dataset

def scenario4_adaptive():
    print("\nLoading dataset (this may take a few seconds)...\n")
    X_chunks, y_chunks = load_dataset("realworld_stream")

    print("\n===== Scenario 4: Adaptive Streaming Learning =====\n")

    sgd = SGDClassifier(loss="log_loss", eta0=0.01, learning_rate="constant", max_iter=5, tol=None)
    asrls = AdaptiveSRLS()
    xgb = XGBClassifier(
        objective="multi:softmax",
        num_class=8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
    )

    X_train_full, y_train_full = np.empty((0, X_chunks[0].shape[1])), np.empty((0,), dtype=int)

    for i, (X_train, y_train) in enumerate(zip(X_chunks, y_chunks), start=1):
        print(f"\n===== Processing Chunk {i:02d} =====")

        # รวมข้อมูลสะสม
        X_train_full = np.vstack([X_train_full, X_train])
        y_train_full = np.concatenate([y_train_full, y_train])

        # normalize classes ให้ต่อเนื่อง
        unique_classes = np.unique(y_train_full)
        class_map = {v: idx for idx, v in enumerate(sorted(unique_classes))}
        y_train_full_mapped = np.array([class_map[y] for y in y_train_full])

        # ============ SGD ============
        try:
            sgd.partial_fit(X_train, y_train, classes=np.arange(len(unique_classes)))
            acc_sgd = sgd.score(X_train, y_train)
        except Exception:
            acc_sgd = 0.0
        print(f"SGD:   acc={acc_sgd:.3f}")

        # ============ A-SRLS ============
        try:
            asrls.fit(X_train, y_train)
            acc_asrls = asrls.score(X_train, y_train)
        except Exception:
            acc_asrls = 0.0
        print(f"A-SRLS: acc={acc_asrls:.3f}")

        # ============ XGB ============
        try:
            xgb.fit(X_train_full, y_train_full_mapped)
            acc_xgb = xgb.score(X_train, [class_map[y] for y in y_train])
        except Exception as e:
            print("XGB Error:", e)
            acc_xgb = 0.0
        print(f"XGB:   acc={acc_xgb:.3f}")

if __name__ == "__main__":
    scenario4_adaptive()
