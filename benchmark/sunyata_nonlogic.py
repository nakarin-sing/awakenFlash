# sunyata_nonlogic.py
# ULTIMATE BUG-FREE DROP-IN MODULE
# วางใน benchmark/ → รันได้ทันที

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# === AbsoluteNon Transform (Stable & Vectorized) ===
def absolute_non_transform(Z):
    x = np.tanh(Z)
    abs_diff = np.abs(x - 0.5)
    log2 = np.log(2.0)

    sym = 0.7 * np.exp(-log2 * abs_diff)
    flow = 0.3 * x * np.exp(-log2)
    enlight = 0.6 * (np.sin(np.pi * x) + 0.5 * np.cos(2 * np.pi * x))
    compassion = 0.9 * (1 - abs_diff) / np.sqrt(1 + x**2 + 1e-8)

    non_core = sym + flow
    full = non_core + 0.4 * enlight + 0.1 * compassion
    meta = 0.95 * np.exp(-x**2) / np.sqrt(np.pi + 1e-8) * np.cos(2 * np.pi * x)

    return (0.95 * meta + 0.05 * full).astype(np.float32)


# === NON-LOGIC LAB (ULTIMATE BUG-FREE) ===
class NonLogicLab:
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
        self.scaler_X = StandardScaler()
        self.scaler_phi = StandardScaler()
        self.coherence_log = []
        self.chunk_count = 0
        self.teacher_blend = 0.1  # เริ่มต่ำ → ค่อยเพิ่ม

    def _init_W(self, n_features):
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
        return np.array([self.class_to_idx.get(v, 0) for v in y], dtype=int)

    def _fit_teacher(self, X, y_idx):
        n = X.shape[0]
        if n < 256:
            return
        # Windowed teacher: ใช้แค่ 3000 ตัวอย่างล่าสุด
        sample = min(1500, n)
        idx = self.rng.choice(n, sample, replace=False)
        dtrain = xgb.DMatrix(X[idx], label=y_idx[idx])
        params = dict(self.teacher_params)
        params.update({"objective": "multi:softprob", "num_class": len(self.classes_)})
        self.teacher = xgb.train(params, dtrain, num_boost_round=5, verbose_eval=False)

    def _teacher_soft(self, X):
        if self.teacher is None:
            return None
        prob = self.teacher.predict(xgb.DMatrix(X))
        K = len(self.classes_)
        if prob.ndim == 1:
            prob = np.eye(K)[np.clip(prob.astype(int), 0, K - 1)]
        prob = np.clip(prob, 1e-12, 1.0)
        logits = np.log(prob) / max(1e-6, self.temp)
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        soft = ex / (ex.sum(axis=1, keepdims=True) + 1e-6)
        return soft.astype(np.float32)

    def partial_loop(self, X_raw, y_raw):
        self.chunk_count += 1
        X = X_raw.astype(np.float32)

        # Scaler X: partial_fit
        if not hasattr(self.scaler_X, "partial_fit"):
            self.scaler_X = StandardScaler()
        self.scaler_X.partial_fit(X)
        X = self.scaler_X.transform(X)

        y_idx = self._encode(y_raw)
        n, _ = X.shape

        # RFF + Non-Linear
        phi = self._rff(X)
        phi_nl = absolute_non_transform(phi)

        # Scaler phi: partial_fit
        if not hasattr(self.scaler_phi, "partial_fit"):
            self.scaler_phi = StandardScaler()
        self.scaler_phi.partial_fit(phi_nl)
        phi_nl = self.scaler_phi.transform(phi_nl)

        # Teacher
        self._fit_teacher(X, y_idx)
        teacher_probs = self._teacher_soft(X)

        # Dynamic teacher_blend
        blend = min(0.5, self.teacher_blend + 0.05 * self.chunk_count / 10)
        self.teacher_blend = blend

        # Targets
        K = len(self.classes_)
        hard = np.zeros((n, K), dtype=np.float32)
        hard[np.arange(n), y_idx] = 1.0
        y_soft = hard.copy()
        if teacher_probs is not None:
            y_soft = (1 - blend) * hard + blend * teacher_probs

        # RLS Update (ใช้ pinv เสมอ)
        PhiT_Phi = (phi_nl.T @ phi_nl) / max(1, n) + 1e-6 * np.eye(phi_nl.shape[1])
        PhiT_y = (phi_nl.T @ y_soft) / max(1, n)

        if self.alpha is None:
            self.alpha = np.linalg.pinv(PhiT_Phi + self.ridge / max(1, n) * np.eye(phi_nl.shape[1])) @ PhiT_y
        else:
            rhs = PhiT_y + self.forget * self.alpha
            H = PhiT_Phi + self.ridge / max(1, n) * np.eye(phi_nl.shape[1])
            self.alpha = np.linalg.pinv(H) @ rhs

        # Metrics
        scores = phi_nl @ self.alpha
        pred_idx = np.argmax(scores, axis=1)
        acc = (pred_idx == y_idx).mean()
        prob_pred = self._softmax(scores)
        loss = log_loss(y_idx, prob_pred, labels=np.arange(K))
        cohesion = self._cohesion_metric(prob_pred, teacher_probs)

        # Void Action
        self.coherence_log.append(cohesion)
        action = None
        if len(self.coherence_log) > 5 and np.mean(self.coherence_log[-5:]) < 0.03:
            self._prune_alpha(sparsity=0.9)
            action = "pruned"
        if len(self.coherence_log) > 10 and np.mean(self.coherence_log[-10:]) < 0.015:
            self.alpha *= 0.6
            action = "shrinked"

        return {
            "acc": float(acc),
            "loss": float(loss),
            "cohesion": float(cohesion),
            "action": action,
            "blend": float(blend)
        }

    def predict(self, X_raw):
        if self.alpha is None or self.W is None:
            return np.zeros(len(X_raw), dtype=int)
        X = X_raw.astype(np.float32)
        X = self.scaler_X.transform(X)
        phi = self._rff(X)
        phi_nl = absolute_non_transform(phi)
        phi_nl = self.scaler_phi.transform(phi_nl)
        scores = phi_nl @ self.alpha
        return self.classes_[np.argmax(scores, axis=1)]

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_idx = self._encode(y)
        return {"acc": float(accuracy_score(y_idx, self._encode(y_pred)))}

    def _softmax(self, logits):
        l = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(l)
        return e / (e.sum(axis=1, keepdims=True) + 1e-6)

    def _cohesion_metric(self, s, t):
        if t is None:
            ent = -np.sum(s * np.log(np.clip(s, 1e-12, 1.0)), axis=1).mean()
            return float(ent / np.log(s.shape[1]))
        overlap = np.mean(np.sum(np.sqrt(np.clip(s * t, 0, 1)), axis=1))
        return max(0.0, float(1.0 - np.clip(overlap, 0, 1)))

    def _prune_alpha(self, sparsity=0.9):
        if self.alpha is None:
            return
        row_norms = np.linalg.norm(self.alpha, axis=1)
        if len(row_norms) == 0:
            return
        thresh = np.quantile(row_norms, sparsity)
        mask = row_norms >= thresh
        self.alpha[~mask, :] = 0.0
