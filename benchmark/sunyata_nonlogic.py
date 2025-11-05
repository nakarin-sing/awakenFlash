# sunyata_nonlogic.py
# DROP-IN MODULE สำหรับ awakenFlash
# วางไฟล์นี้ใน benchmark/ หรือ awakenFlash/
# ใช้: from sunyata_nonlogic import NonLogicLab

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# === Lightweight AbsoluteNon Transform (Vectorized & Stable) ===
def absolute_non_transform(Z, alpha=0.7, beta=0.6, gamma=0.95, delta=0.9):
    """
    Z: (n_samples, D) → float32
    Returns: (n_samples, D) non-linear transformed
    """
    x = np.tanh(Z)  # [-1, 1]
    abs_diff = np.abs(x - 0.5)
    log2 = np.log(2.0)

    sym = alpha * np.exp(-log2 * abs_diff)
    flow = (1 - alpha) * x * np.exp(-log2)
    enlight = beta * (np.sin(np.pi * x) + 0.5 * np.cos(2 * np.pi * x))
    compassion = delta * (1 - abs_diff) / np.sqrt(1 + x**2 + 1e-8)
    linear = 0.5 + (x - 0.5) * np.exp(-log2)

    non_core = sym + flow + 1e-12 * linear
    full = non_core + (1 - beta) * enlight + (1 - delta) * compassion
    meta = gamma * np.exp(-x**2) / np.sqrt(np.pi + 1e-8) * np.cos(2 * np.pi * x)

    return (gamma * meta + (1 - gamma) * full).astype(np.float32)


# === NON-LOGIC LAB (3 Worlds: Logic, Mind, Void) ===
class NonLogicLab:
    """
    Drop-in streaming learner.
    Methods:
      - partial_loop(X, y) → dict(metrics)
      - predict(X) → np.array
      - evaluate(X, y) → dict(acc)
    """
    def __init__(self, D_rff=512, ridge=25.0, forget=0.995, temp=2.0, seed=42):
        self.rng = np.random.default_rng(seed)
        self.D = int(D_rff)
        self.ridge = float(ridge)
        self.forget = float(forget)
        self.temp = float(temp)
        self.W = None
        self.alpha = None
        self.classes_ = None
        self.class_to_idx = {}
        self.teacher = None
        self.teacher_params = {"max_depth": 3, "eta": 0.3, "verbosity": 0}
        self.scaler = StandardScaler()
        self.coherence_log = []

    def _init_W(self, n_features):
        if self.W is None or self.W.shape[1] != n_features:
            rows = self.D // 2
            self.W = self.rng.normal(0, 1.0 / np.sqrt(n_features), (rows, n_features)).astype(np.float32)

    def _rff(self, X):
        self._init_W(X.shape[1])
        proj = X.astype(np.float32) @ self.W.T
        phi = np.hstack([np.cos(proj), np.sin(proj)]) * np.sqrt(2.0 / self.D)
        return phi

    def _encode(self, y):
        if self.classes_ is None:
            self.classes_ = np.unique(y)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self.class_to_idx.get(v, -1) for v in y], dtype=int)

    def _fit_teacher(self, X, y_idx):
        n = X.shape[0]
        if self.teacher is None and n >= 256:
            sample = min(2000, n)
            idx = self.rng.choice(n, sample, replace=False)
            dtrain = xgb.DMatrix(X[idx], label=y_idx[idx])
            params = dict(self.teacher_params)
            params.update({"objective": "multi:softprob", "num_class": len(self.classes_)})
            self.teacher = xgb.train(params, dtrain, num_boost_round=6, verbose_eval=False)

    def _teacher_soft(self, X):
        if self.teacher is None:
            return None
        prob = self.teacher.predict(xgb.DMatrix(X))
        if prob.ndim == 1:
            K = len(self.classes_)
            prob = np.eye(K)[np.clip(prob.astype(int), 0, K - 1)]
        # Temperature scaling
        logits = np.log(np.clip(prob, 1e-12, 1.0)) / max(1e-6, self.temp)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        soft = ex / ex.sum(axis=1, keepdims=True)
        return soft.astype(np.float32)

    def partial_loop(self, X_raw, y_raw, teacher_blend=0.4):
        X = X_raw.astype(np.float32)
        if hasattr(self.scaler, "mean_"):
            X = self.scaler.transform(X)
        else:
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        y_idx = self._encode(y_raw)
        if np.any(y_idx == -1):
            raise ValueError("Unknown class in y_raw")

        n, _ = X.shape
        phi = self._rff(X)
        phi_nl = absolute_non_transform(phi)

        # Teacher
        self._fit_teacher(X, y_idx)
        teacher_probs = self._teacher_soft(X)

        # Targets
        K = len(self.classes_)
        hard = np.zeros((n, K), dtype=np.float32)
        hard[np.arange(n), y_idx] = 1.0
        y_soft = hard.copy()
        if teacher_probs is not None:
            y_soft = (1 - teacher_blend) * hard + teacher_blend * teacher_probs

        # RLS Update
        PhiT_Phi = (phi_nl.T @ phi_nl) / max(1, n)
        PhiT_y = (phi_nl.T @ y_soft) / max(1, n)
        H = PhiT_Phi + np.eye(PhiT_Phi.shape[0]) * (self.ridge / max(1, n))

        if self.alpha is None:
            self.alpha = np.linalg.solve(H + 1e-6 * np.eye(H.shape[0]), PhiT_y)
        else:
            rhs = PhiT_y + self.forget * self.alpha
            try:
                self.alpha = np.linalg.solve(H + 1e-6 * np.eye(H.shape[0]), rhs)
            except np.linalg.LinAlgError:
                self.alpha = np.linalg.pinv(H + 1e-6 * np.eye(H.shape[0])) @ rhs

        # Metrics
        scores = phi_nl @ self.alpha
        pred_idx = np.argmax(scores, axis=1)
        acc = (pred_idx == y_idx).mean()
        loss = log_loss(y_soft, self._softmax(scores))
        cohesion = self._cohesion_metric(scores, teacher_probs)

        # Void Action
        self.coherence_log.append(cohesion)
        action = None
        if len(self.coherence_log) > 5 and np.mean(self.coherence_log[-5:]) < 0.04:
            self._prune_alpha(sparsity=0.7)
            action = "pruned"
        if len(self.coherence_log) > 10 and np.mean(self.coherence_log[-10:]) < 0.02:
            self.alpha *= 0.5
            action = "shrinked"

        return {
            "acc": float(acc),
            "loss": float(loss),
            "cohesion": float(cohesion),
            "action": action
        }

    def predict(self, X_raw):
        if self.alpha is None:
            return np.zeros(len(X_raw), dtype=int)
        X = X_raw.astype(np.float32)
        X = self.scaler.transform(X)
        phi = self._rff(X)
        phi_nl = absolute_non_transform(phi)
        scores = phi_nl @ self.alpha
        return self.classes_[np.argmax(scores, axis=1)]

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {"acc": float(accuracy_score(y, y_pred))}

    def _softmax(self, logits):
        l = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(l)
        return e / (e.sum(axis=1, keepdims=True) + 1e-8)

    def _cohesion_metric(self, scores, teacher_probs):
        s = self._softmax(scores)
        if teacher_probs is None:
            ent = -np.sum(s * np.log(np.clip(s, 1e-12, 1.0)), axis=1).mean()
            return float(ent / np.log(s.shape[1]))
        overlap = np.mean(np.sum(np.sqrt(s * teacher_probs), axis=1))
        return max(0.0, float(1.0 - overlap))

    def _prune_alpha(self, sparsity=0.5):
        if self.alpha is None:
            return
        row_norms = np.linalg.norm(self.alpha, axis=1)
        if len(row_norms) == 0:
            return
        thresh = np.quantile(row_norms, sparsity)
        mask = row_norms >= thresh
        self.alpha[~mask, :] = 0.0
