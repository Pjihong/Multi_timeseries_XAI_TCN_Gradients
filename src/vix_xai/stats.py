"""
stats.py — 통계 검정 모듈.

각 함수가 어떤 가설에 매핑되는지 명시.

H1: two_sample_perm        — event vs calm DTW (비쌍체)
H2: paired_bootstrap_auc   — AUC(CAM-DTW) vs AUC(Raw DTW)
H3: alignment_perm         — CAM-market alignment > random
H4: paired_perm + deletion — important vs random deletion (쌍체)
H5: accuracy_above_chance  — TCAV CV accuracy > 0.5
H6: two_sample_perm        — CWCR event vs calm (비쌍체)
H7: interaction_perm       — event × concept interaction
H8: cosine_stability       — CAV fold간 stability
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


# ═══════════════════════════════════════════════════════════════════
# Subsampling
# ═══════════════════════════════════════════════════════════════════

def subsample_nonoverlap(indices: np.ndarray, min_gap: int) -> np.ndarray:
    """sliding window 겹침 제거. min_gap >= seq_len 권장."""
    if len(indices) == 0:
        return indices
    indices = np.sort(indices)
    kept = [indices[0]]
    for idx in indices[1:]:
        if idx - kept[-1] >= min_gap:
            kept.append(idx)
    return np.array(kept, dtype=int)


# ═══════════════════════════════════════════════════════════════════
# H1, H6, H6b: Two-sample permutation (비쌍체, label shuffle)
# ═══════════════════════════════════════════════════════════════════

def two_sample_perm(
    a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 42,
) -> dict:
    """
    비쌍체 two-sample permutation test.
    H0: mean(a) = mean(b). Null: 합친 후 label shuffle.

    event DTW vs calm DTW처럼 자연적 쌍이 없는 두 그룹 비교.
    """
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    pooled = np.concatenate([a, b])
    na = len(a)
    obs = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(pooled[p[:na]].mean() - pooled[p[na:]].mean()) >= abs(obs):
            count += 1
    return dict(
        mean_diff=obs,
        p_value=count / n_perm,
        effect_size=obs / (pooled.std() + 1e-12),
        n_a=len(a), n_b=len(b),
    )


# ═══════════════════════════════════════════════════════════════════
# H2: Paired bootstrap AUC comparison
# ═══════════════════════════════════════════════════════════════════

def paired_bootstrap_auc(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
    n_boot: int = 2000, alpha: float = 0.05, seed: int = 42,
) -> dict:
    """
    동일 부트스트랩 샘플에서 AUC_a - AUC_b의 CI.
    H0: AUC_a <= AUC_b.
    """
    y = np.asarray(y_true).ravel()
    sa, sb = np.asarray(scores_a).ravel(), np.asarray(scores_b).ravel()
    n = len(y)
    if len(np.unique(y)) < 2:
        return dict(auc_a=np.nan, auc_b=np.nan, delta=np.nan,
                    ci_low=np.nan, ci_high=np.nan, p_value=np.nan)
    auc_a, auc_b = roc_auc_score(y, sa), roc_auc_score(y, sb)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            deltas.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
        except Exception:
            pass
    if len(deltas) < 10:
        return dict(auc_a=float(auc_a), auc_b=float(auc_b),
                    delta=float(auc_a - auc_b),
                    ci_low=np.nan, ci_high=np.nan, p_value=np.nan)
    deltas = np.array(deltas)
    return dict(
        auc_a=float(auc_a), auc_b=float(auc_b),
        delta=float(auc_a - auc_b),
        ci_low=float(np.percentile(deltas, 100 * alpha / 2)),
        ci_high=float(np.percentile(deltas, 100 * (1 - alpha / 2))),
        p_value=float(np.mean(deltas <= 0)),
    )


# ═══════════════════════════════════════════════════════════════════
# H3: Within-window permutation for CAM-market alignment
# ═══════════════════════════════════════════════════════════════════

def alignment_perm(
    cams: np.ndarray, raw_windows: np.ndarray,
    top_k: int = 3, tolerance: int = 1,
    n_perm: int = 1000, seed: int = 42,
) -> dict:
    """
    CAM peaks가 market movement peaks와 random보다 잘 맞는지.
    Null: 각 window 내에서 CAM을 shuffle.
    """
    N = len(cams)
    rng = np.random.default_rng(seed)

    def _overlap(cam, raw):
        T = len(cam)
        ch = np.zeros(T)
        ch[1:] = np.abs(np.diff(raw))
        cp = np.argsort(-np.abs(cam))[:top_k]
        mp = np.argsort(-ch)[:top_k]
        return sum(1 for c in cp if any(abs(c - m) <= tolerance for m in mp)) / top_k

    real = np.array([_overlap(cams[i], raw_windows[i]) for i in range(N)])
    real_mean = float(real.mean())
    null_means = []
    for _ in range(n_perm):
        nm = np.mean([_overlap(rng.permutation(cams[i]), raw_windows[i]) for i in range(N)])
        null_means.append(nm)
    null_means = np.array(null_means)
    return dict(
        real_mean=real_mean,
        null_mean=float(null_means.mean()),
        p_value=float(np.mean(null_means >= real_mean)),
        effect_size=float((real_mean - null_means.mean()) / (null_means.std() + 1e-12)),
        n=N,
    )


# ═══════════════════════════════════════════════════════════════════
# H4: Paired permutation (sign-flip) — for deletion test
# ═══════════════════════════════════════════════════════════════════

def paired_perm(
    a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 42,
) -> dict:
    """
    쌍체 sign-flip permutation. H0: mean(a - b) = 0.
    같은 샘플에서 important vs random deletion 비교 등.
    """
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    obs = float(diff.mean())
    rng = np.random.default_rng(seed)
    count = sum(
        1 for _ in range(n_perm)
        if abs((diff * rng.choice([-1, 1], n)).mean()) >= abs(obs)
    )
    return dict(
        mean_diff=obs,
        p_value=count / n_perm,
        effect_size=obs / (diff.std() + 1e-12),
        n=n,
    )


def deletion_test(
    model, X_batch, cam_batch, mask_pct=90, n_random=100, device="cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CAM-important timestep 삭제 vs random 삭제.
    Returns (important_delta, random_mean_delta) per sample.
    """
    import torch
    model.eval()
    X = X_batch.to(device)
    with torch.no_grad():
        y0 = model(X).squeeze(-1).cpu().numpy()
    fill = X.mean(dim=[0, 1], keepdim=True)
    N, T, F = X.shape
    Xm = X.clone()
    for i in range(N):
        thr = np.percentile(cam_batch[i], mask_pct)
        Xm[i, cam_batch[i] >= thr, :] = fill[0, 0, :]
    with torch.no_grad():
        y_imp = model(Xm).squeeze(-1).cpu().numpy()
    imp_d = np.abs(y0 - y_imp)
    n_mask = max(1, int(T * (100 - mask_pct) / 100))
    rng = np.random.default_rng(42)
    rds = []
    for _ in range(n_random):
        Xr = X.clone()
        for i in range(N):
            Xr[i, rng.choice(T, n_mask, replace=False), :] = fill[0, 0, :]
        with torch.no_grad():
            rds.append(np.abs(y0 - model(Xr).squeeze(-1).cpu().numpy()))
    return imp_d, np.stack(rds).mean(axis=0)


# ═══════════════════════════════════════════════════════════════════
# H5: TCAV accuracy > chance
# ═══════════════════════════════════════════════════════════════════

def accuracy_above_chance(
    accuracies: np.ndarray, chance: float = 0.5,
    n_perm: int = 5000, seed: int = 42,
) -> dict:
    """
    CV accuracy가 chance보다 유의하게 높은지.
    Null: U(0,1) 에서 추출한 accuracy의 mean 분포.
    """
    accs = np.asarray(accuracies).ravel()
    obs = float(accs.mean())
    rng = np.random.default_rng(seed)
    null = [rng.random(len(accs)).mean() for _ in range(n_perm)]
    return dict(
        mean_acc=obs, chance=chance,
        p_value=float(np.mean(np.array(null) >= obs)),
        n_folds=len(accs),
    )


# ═══════════════════════════════════════════════════════════════════
# H7: Interaction contrast (event × concept)
# ═══════════════════════════════════════════════════════════════════

def interaction_perm(
    scores: np.ndarray, is_event: np.ndarray, is_concept: np.ndarray,
    n_perm: int = 5000, seed: int = 42,
) -> dict:
    """
    2×2 interaction:
    contrast = (ev&co) - (ev&!co) - (!ev&co) + (!ev&!co)
    Null: concept labels shuffle.
    """
    s = np.asarray(scores).ravel()
    ev = np.asarray(is_event).ravel().astype(bool)
    co = np.asarray(is_concept).ravel().astype(bool)

    def _c(s_, e_, c_):
        g = [s_[e_ & c_], s_[e_ & ~c_], s_[~e_ & c_], s_[~e_ & ~c_]]
        if any(len(x) == 0 for x in g):
            return np.nan
        return g[0].mean() - g[1].mean() - g[2].mean() + g[3].mean()

    obs = _c(s, ev, co)
    if np.isnan(obs):
        return dict(interaction=np.nan, p_value=np.nan, effect_size=np.nan)
    rng = np.random.default_rng(seed)
    count = sum(1 for _ in range(n_perm)
                if (v := _c(s, ev, rng.permutation(co))) is not np.nan and abs(v) >= abs(obs))
    return dict(
        interaction=float(obs),
        p_value=count / n_perm,
        effect_size=float(obs / (s.std() + 1e-12)),
    )


# ═══════════════════════════════════════════════════════════════════
# H8: CAV stability
# ═══════════════════════════════════════════════════════════════════

def cosine_stability(fold_cavs: list, threshold: float = 0.8) -> dict:
    """Fold간 CAV cosine similarity."""
    sims = [float(np.dot(fold_cavs[i], fold_cavs[j]))
            for i in range(len(fold_cavs))
            for j in range(i + 1, len(fold_cavs))]
    sims = np.array(sims) if sims else np.array([np.nan])
    return dict(
        mean_cos=float(np.nanmean(sims)),
        min_cos=float(np.nanmin(sims)),
        all_above=bool(np.all(sims >= threshold)),
        n_pairs=len(sims),
    )


# ═══════════════════════════════════════════════════════════════════
# Block bootstrap CI + FDR
# ═══════════════════════════════════════════════════════════════════

def block_bootstrap_ci(
    values: np.ndarray, block_len: int = 20,
    n_boot: int = 2000, alpha: float = 0.05, seed: int = 42,
) -> dict:
    """Block bootstrap CI for dependent data."""
    v = np.asarray(values).ravel()
    n = len(v)
    if n == 0:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan)
    rng = np.random.default_rng(seed)
    mx = max(1, n - block_len + 1)
    nb = max(1, n // block_len)
    means = [np.concatenate([v[s:s + block_len] for s in rng.integers(0, mx, nb)])[:n].mean()
             for _ in range(n_boot)]
    return dict(
        mean=float(v.mean()),
        ci_low=float(np.percentile(means, 100 * alpha / 2)),
        ci_high=float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """FDR correction. Guarantees corrected[i] >= pvals[i]."""
    p = np.asarray(pvals, np.float64)
    n = len(p)
    if n == 0:
        return p.copy()
    # sort p-values
    si = np.argsort(p)
    sp = p[si]
    # BH formula: p_i * n / rank_i
    corrected = sp * n / np.arange(1, n + 1)
    # enforce monotonicity: from largest rank to smallest
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)
    # put back in original order
    result = np.empty(n)
    result[si] = corrected
    return result