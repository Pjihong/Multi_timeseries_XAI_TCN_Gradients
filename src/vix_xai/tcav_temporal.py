"""
tcav_temporal.py — Temporal TCAV: CAV learning + directional derivative + scoring.

역할: TCAV 엔진만. 통계/플롯 없음.

주력 지표:
  CWCR(i) = Σ_t [ cam̃_t × max(dd_t, 0) ]
  where dd_t = ⟨dY/dE_t, v_c⟩
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class TemporalTCAV:
    """
    Temporal TCAV with multiple pooling strategies.

    Pooling (CAV 학습 시):
      "mean"         — 시간축 단순 평균 (baseline)
      "cam_weighted" — CAM 가중 평균 (PRIMARY)
      "segment"      — early/mid/late concat

    Parameters
    ----------
    pooling : str
    C : float — regularization
    cv_folds : int
    seed : int
    """

    def __init__(
        self,
        pooling: str = "cam_weighted",
        C: float = 1.0,
        cv_folds: int = 5,
        seed: int = 42,
        n_segments: int = 3,
    ):
        assert pooling in ("mean", "cam_weighted", "segment")
        self.pooling = pooling
        self.C = C
        self.cv_folds = cv_folds
        self.seed = seed
        self.n_segments = n_segments
        self.clf_: Optional[LogisticRegression] = None
        self.scaler_: Optional[StandardScaler] = None
        self.cv_results_: List[dict] = []
        self.fold_cavs_: List[np.ndarray] = []

    # ── Pooling ────────────────────────────────────────────────

    def _pool(self, E: np.ndarray, cam: Optional[np.ndarray] = None) -> np.ndarray:
        """(N,T,C) → (N,D)."""
        N, T, C = E.shape
        if self.pooling == "mean":
            return E.mean(axis=1)
        elif self.pooling == "cam_weighted":
            if cam is None:
                raise ValueError("cam required for cam_weighted")
            w = np.clip(cam, 0, None)
            w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
            return (E * w[:, :, np.newaxis]).sum(axis=1)
        elif self.pooling == "segment":
            seg = T // self.n_segments
            parts = []
            for s in range(self.n_segments):
                lo = s * seg
                hi = lo + seg if s < self.n_segments - 1 else T
                parts.append(E[:, lo:hi, :].mean(axis=1))
            return np.hstack(parts)
        raise ValueError(f"Unknown pooling: {self.pooling}")

    # ── Fit ────────────────────────────────────────────────────

    def fit(
        self,
        E_pos: np.ndarray,
        E_neg: np.ndarray,
        cam_pos: Optional[np.ndarray] = None,
        cam_neg: Optional[np.ndarray] = None,
    ) -> "TemporalTCAV":
        """
        CAV 학습.

        Parameters
        ----------
        E_pos, E_neg : (N, T, C) embeddings
        cam_pos, cam_neg : (N, T) CAM (cam_weighted 시 필요)
        """
        X = np.vstack([self._pool(E_pos, cam_pos), self._pool(E_neg, cam_neg)])
        y = np.array([1] * len(E_pos) + [0] * len(E_neg))
        assert len(np.unique(y)) >= 2

        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)

        skf = StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.seed)
        self.cv_results_, self.fold_cavs_ = [], []

        for fold, (tr, va) in enumerate(skf.split(Xs, y)):
            clf = LogisticRegression(
                C=self.C, max_iter=5000, solver="liblinear",
                class_weight="balanced", random_state=self.seed,
            )
            clf.fit(Xs[tr], y[tr])
            pred, prob = clf.predict(Xs[va]), clf.predict_proba(Xs[va])[:, 1]
            acc = accuracy_score(y[va], pred)
            try:
                auc = roc_auc_score(y[va], prob)
            except Exception:
                auc = float("nan")
            w = clf.coef_.ravel() / (self.scaler_.scale_ + 1e-12)
            w = w / (np.linalg.norm(w) + 1e-12)
            self.fold_cavs_.append(w)
            self.cv_results_.append(dict(fold=fold, accuracy=acc, roc_auc=auc))

        self.clf_ = LogisticRegression(
            C=self.C, max_iter=5000, solver="liblinear",
            class_weight="balanced", random_state=self.seed,
        )
        self.clf_.fit(Xs, y)
        return self

    def get_cav(self) -> np.ndarray:
        """CAV (unit norm, original scale)."""
        w = self.clf_.coef_.ravel() / (self.scaler_.scale_ + 1e-12)
        return w / (np.linalg.norm(w) + 1e-12)

    def get_cv_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.cv_results_)

    # ── Directional derivative ─────────────────────────────────

    def directional_derivative(
        self, dYdE: np.ndarray, v_c: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        dd_t(i) = ⟨dY/dE_t(i), v_c⟩

        Parameters
        ----------
        dYdE : (N, T, C)
        v_c  : (C,) — None이면 self.get_cav()

        Returns
        -------
        dd : (N, T)
        """
        if v_c is None:
            v_c = self.get_cav()
        v_c = np.asarray(v_c, np.float64).ravel()
        C = dYdE.shape[2]
        if v_c.shape[0] != C:
            raise ValueError(
                f"CAV dim {v_c.shape[0]} != embedding dim {C}. "
                "segment pooling CAV는 dd에 사용 불가."
            )
        return dYdE @ v_c

    # ── Score aggregation ──────────────────────────────────────

    @staticmethod
    def aggregate(
        dd: np.ndarray, cam: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        dd_t → sample-level scores.

        Returns dict:
          sign_score            : mean(dd_t > 0)
          positive_mass         : mean(max(dd_t, 0))
          cam_weighted_pos_mass : Σ_t cam̃_t × max(dd_t, 0)  ← PRIMARY
          mean_dd               : mean(dd_t)
        """
        dd_pos = np.clip(dd, 0, None)
        scores: Dict[str, np.ndarray] = {
            "sign_score": (dd > 0).astype(float).mean(axis=1),
            "positive_mass": dd_pos.mean(axis=1),
            "mean_dd": dd.mean(axis=1),
        }
        if cam is not None:
            w = np.clip(cam, 0, None)
            w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
            scores["cam_weighted_pos_mass"] = (w * dd_pos).sum(axis=1)
        else:
            scores["cam_weighted_pos_mass"] = dd_pos.mean(axis=1)
        return scores
