"""
analysis.py — GradCAM-centric analysis pipeline.

entry point: run_analysis(...)

가설 검정:
  H1: CAM-DTW(event) ≠ CAM-DTW(calm)        — stats.two_sample_perm
  H2: AUC(CAM-DTW) > AUC(Raw DTW)           — stats.paired_bootstrap_auc
  H3: CAM peaks ↔ market > random            — stats.alignment_perm
  H4: CAM deletion > random deletion         — stats.paired_perm + deletion_test
  H5: TCAV accuracy > 0.5                    — stats.accuracy_above_chance
  H6: CWCR(event) ≠ CWCR(calm)             — stats.two_sample_perm
  H6b: CWCR(concept-on) ≠ CWCR(off)        — stats.two_sample_perm
  H7: event × concept interaction            — stats.interaction_perm
  H8: CAV fold stability                     — stats.cosine_stability
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import event_wraping as ew
from .gradcam import (
    TimeSeriesGradCAM,
    TemporalGradientExtractor,
    extract_embeddings,
    resolve_target_branch,
)
from .tcav_temporal import TemporalTCAV
from . import stats


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _collect(dl: DataLoader) -> torch.Tensor:
    return torch.cat([x for x, _ in dl], dim=0)


def _define_events(
    level, horizon=5, q_event=0.95, q_ctrl=(0.4, 0.6),
    positive_only=True, exclude_buf=3,
):
    """positive spike event 정의. Returns (fm, ev_mask, ctrl_mask)."""
    level = np.asarray(level, np.float64).ravel()
    n = len(level)
    fm = np.full(n, np.nan)
    for i in range(n - horizon):
        fut = level[i + 1 : i + 1 + horizon]
        fm[i] = (fut - level[i]).max() if positive_only else np.abs(fut - level[i]).max()
    valid = fm[np.isfinite(fm)]
    if len(valid) == 0:
        return fm, np.zeros(n, bool), np.zeros(n, bool)
    thr = np.quantile(valid, q_event)
    lo, hi = np.quantile(valid, q_ctrl[0]), np.quantile(valid, q_ctrl[1])
    ev = fm >= thr
    ctrl = (fm >= lo) & (fm <= hi)
    if exclude_buf > 0:
        for i in np.where(ev)[0]:
            ctrl[max(0, i - exclude_buf) : min(n, i + exclude_buf + 1)] = False
    return fm, ev, ctrl


def _concept_labels(df_raw, train_end, te_index, seq_len, N, q_safe=0.9):
    """Flight-to-Safety concept labels."""
    def _res(df, cands):
        for c in (cands if isinstance(cands, (list, tuple)) else [cands]):
            if c in df.columns:
                return c
        norm = lambda s: re.sub(r"[\s\-\_\.]+", "", str(s).lower())
        m = {norm(c): c for c in df.columns}
        for c in (cands if isinstance(cands, (list, tuple)) else [cands]):
            if norm(c) in m:
                return m[norm(c)]
        raise KeyError(f"Cannot resolve {cands}")

    df = df_raw.copy()
    te_index = pd.Index(pd.to_datetime(te_index))
    try:
        gold = _res(df, ["Gold"])
        risk = _res(df, ["SPX", "S&P"])
    except KeyError:
        return np.zeros(N, dtype=np.int8)
    sr = df[gold].pct_change()
    rr = df[risk].pct_change()
    thr = float(np.nanquantile(sr[df.index <= pd.to_datetime(train_end)].dropna().values, q_safe))
    c = (sr >= thr) & (rr < 0)
    return c.reindex(te_index).iloc[seq_len:].to_numpy(dtype=np.int8)[:N]


def _sfig(fig, path, show=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# DTW distance computation (Path A)
# ═══════════════════════════════════════════════════════════════════

def _dtw_distances(raw_w, Ea, cam_a, ref, N, band=5, alpha=1.5, top_p=0.2):
    rows = []
    for i in tqdm(range(N), desc="DTW", unit="w"):
        j = ref if ref != i else max(0, i - 1)
        C_raw = np.abs(raw_w[i][:, None] - raw_w[j][None, :])
        raw_d = float(ew.dtw_from_cost_matrix(C_raw, band=band, normalize=True).normalized_cost)
        emb_d = float(ew.dtdw_embedding(Ea[i], Ea[j], method="l2", k=0, band=band).normalized_cost)
        ecam_d = float(ew.wdtdw_embedding(
            Ea[i], Ea[j], g_a=cam_a[i], g_b=cam_a[j],
            emb_method="l2", emb_k=0, band=band, normalize=True,
            alpha=alpha, top_p=top_p, weight_mode="local",
        ).normalized_cost)
        rows.append(dict(idx=i, raw_dtw=raw_d, emb_dtw=emb_d, emb_cam_dtw=ecam_d))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_analysis(
    *,
    model: torch.nn.Module,
    meta: dict,
    cfg,
    df_raw: pd.DataFrame,
    dl_te: DataLoader,
    device: torch.device,
    target_col: str = "VIX",
    target_branch: Optional[int] = None,
    horizon: int = 5,
    q_event: float = 0.95,
    positive_only: bool = True,
    min_gap: int = 20,
    smooth_steps: int = 0,
    run_path_a: bool = True,
    band: int = 5,
    alpha: float = 1.5,
    top_p: float = 0.2,
    run_path_b: bool = True,
    pooling: str = "cam_weighted",
    tcav_folds: int = 5,
    n_perm: int = 5000,
    n_boot: int = 2000,
    save_dir: str = "outputs/analysis",
    show: bool = False,
    seed: int = 42,
) -> dict:
    """
    GradCAM-centric analysis pipeline.

    Usage:
    >>> result = run_analysis(model=model, meta=meta, cfg=cfg,
    ...     df_raw=df_raw, dl_te=dl_te, device=device)
    >>> print(result["test_df"])
    """
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)
    if target_branch is not None:
        target_idx = target_branch
    else:
        target_idx = int(meta.get("target_index", 0))
    print(f"[Analysis] target branch: {target_idx} "
          f"(features[{target_idx}]={meta.get('feature_names', ['?'])[target_idx]})")

    # ── windows ───────────────────────────────────────────────
    X_all = _collect(dl_te).cpu()
    te_idx = meta["df_te"].index
    seq = cfg.seq_len
    dates = np.array(te_idx[seq:])
    N = min(len(X_all), len(dates))
    X_all, dates = X_all[:N], dates[:N]
    raw_lv = np.nan_to_num(df_raw.reindex(te_idx)[target_col].to_numpy(np.float64))
    raw_w = np.stack([raw_lv[i : i + seq] for i in range(N)])

    # ── events ────────────────────────────────────────────────
    _, ev_m, ct_m = _define_events(raw_lv, horizon, q_event, positive_only=positive_only)
    is_ev = np.zeros(N, np.int8)
    is_ct = np.zeros(N, np.int8)
    for i in range(N):
        d = seq + i
        if d < len(ev_m):
            is_ev[i] = int(ev_m[d])
            is_ct[i] = int(ct_m[d])
    ev_i = stats.subsample_nonoverlap(np.where(is_ev == 1)[0], min_gap)
    ct_i = stats.subsample_nonoverlap(np.where(is_ct == 1)[0], min_gap)
    print(f"[Analysis] N={N} ev={len(ev_i)} ctrl={len(ct_i)}")
    if len(ev_i) < 3 or len(ct_i) < 3:
        print("[Analysis] Not enough samples"); return None

    # ── Target branch CAM ─────────────────────────────────────
    print("[Analysis] Target branch CAM...")
    eng = TimeSeriesGradCAM(model, device)
    cam_s, cam_a = eng.generate_batch(X_all, target_idx, smooth_steps, desc="CAM")
    eng.cleanup()

    # ── Target branch embeddings ──────────────────────────────
    print("[Analysis] Embeddings...")
    Ea = extract_embeddings(model, X_all, target_idx)

    # ── Save ──────────────────────────────────────────────────
    json.dump(dict(target=target_idx, N=N, n_ev=len(ev_i), n_ct=len(ct_i),
                   horizon=horizon, q_event=q_event, min_gap=min_gap,
                   band=band, alpha=alpha, top_p=top_p, pooling=pooling,
                   n_perm=n_perm, seed=seed),
              open(os.path.join(save_dir, "config.json"), "w"), indent=2)
    pd.DataFrame({"date": dates, "is_event": is_ev, "is_control": is_ct}
                 ).to_csv(os.path.join(save_dir, "events.csv"), index=False)

    # ── Mean CAM plot ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, lbl, c in [(ev_i, "event", "tab:red"), (ct_i, "calm", "tab:blue")]:
        m = cam_a[idx].mean(0); se = cam_a[idx].std(0) / np.sqrt(len(idx))
        ax.plot(m, label=f"{lbl}(n={len(idx)})", color=c)
        ax.fill_between(range(len(m)), m - se, m + se, alpha=.15, color=c)
    ax.set_xlabel("Timestep"); ax.set_ylabel("|CAM|")
    ax.set_title("Target Branch CAM: Event vs Calm"); ax.legend(); ax.grid(alpha=.2)
    plt.tight_layout(); _sfig(fig, os.path.join(save_dir, "mean_cam.png"), show)

    # ══════════════════════════════════════════════════════════
    # PATH A: GradCAM + DTW  →  H1, H2, H3, H4
    # ══════════════════════════════════════════════════════════
    test_rows = []
    dtw_df = None
    if run_path_a:
        pd_a = os.path.join(save_dir, "path_a"); os.makedirs(pd_a, exist_ok=True)
        print("[Analysis] Path A...")
        ref = int(np.argmin(np.abs(raw_w.mean(1) - np.median(raw_w.mean(1)))))
        dtw_df = _dtw_distances(raw_w, Ea, cam_a, ref, N, band, alpha, top_p)
        dtw_df["is_event"] = is_ev; dtw_df["date"] = dates
        dtw_df.to_csv(os.path.join(pd_a, "dtw.csv"), index=False)

        # H1
        for mc in ["raw_dtw", "emb_dtw", "emb_cam_dtw"]:
            ev_v = dtw_df.loc[is_ev == 1, mc].values
            ne_v = dtw_df.loc[is_ev == 0, mc].values
            if len(ev_v) > 2 and len(ne_v) > 2:
                r = stats.two_sample_perm(ev_v, ne_v, n_perm, seed)
                test_rows.append(dict(hyp="H1", metric=mc, stat=r["mean_diff"],
                                      p=r["p_value"], es=r["effect_size"],
                                      n=r["n_a"] + r["n_b"]))

        # H2
        y = dtw_df["is_event"].values
        if len(np.unique(y)) >= 2:
            r = stats.paired_bootstrap_auc(
                y, dtw_df["emb_cam_dtw"].values, dtw_df["raw_dtw"].values, n_boot, seed=seed)
            test_rows.append(dict(hyp="H2", metric="emb_cam_vs_raw",
                                  stat=r["delta"], p=r["p_value"], es=r["delta"],
                                  n=len(y),
                                  detail=f"auc_cam={r['auc_a']:.3f} auc_raw={r['auc_b']:.3f} "
                                         f"ci=[{r['ci_low']:.3f},{r['ci_high']:.3f}]"))

        # H3
        print("[Analysis] H3 alignment...")
        r = stats.alignment_perm(cam_a, raw_w, n_perm=min(n_perm, 1000), seed=seed)
        test_rows.append(dict(hyp="H3", metric="cam_market_overlap",
                              stat=r["real_mean"], p=r["p_value"], es=r["effect_size"], n=r["n"],
                              detail=f"real={r['real_mean']:.3f} null={r['null_mean']:.3f}"))

        # H4
        print("[Analysis] H4 deletion...")
        for grp, idx in [("event", ev_i), ("calm", ct_i)]:
            s = idx[:min(100, len(idx))]
            if len(s) < 3: continue
            imp, rnd = stats.deletion_test(model, X_all[s], cam_a[s], device=device)
            r = stats.paired_perm(imp, rnd, n_perm, seed)
            test_rows.append(dict(hyp="H4", metric=f"del_{grp}",
                                  stat=r["mean_diff"], p=r["p_value"],
                                  es=r["effect_size"], n=r["n"]))

        # boxplot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, mc, lbl in zip(axes, ["raw_dtw", "emb_dtw", "emb_cam_dtw"],
                                ["Raw", "Emb", "CAM-Emb"]):
            ax.boxplot([dtw_df.loc[is_ev == 0, mc], dtw_df.loc[is_ev == 1, mc]],
                       labels=["calm", "event"])
            ax.set_title(lbl); ax.grid(alpha=.2, axis="y")
        plt.suptitle("DTW Distances"); plt.tight_layout()
        _sfig(fig, os.path.join(pd_a, "boxplots.png"), show)

    # ══════════════════════════════════════════════════════════
    # PATH B: GradCAM + Temporal TCAV  →  H5, H6, H6b, H7, H8
    # ══════════════════════════════════════════════════════════
    tcav_res = None
    if run_path_b:
        pd_b = os.path.join(save_dir, "path_b"); os.makedirs(pd_b, exist_ok=True)
        print("[Analysis] Path B...")

        print("[Analysis] dY/dE_t...")
        gex = TemporalGradientExtractor(model, device, target_idx)
        _, dYdE = gex.extract_batch(X_all, desc="dY/dE_t")
        gex.cleanup()

        train_end = str(meta["df_tr"].index[-1])
        cl = _concept_labels(df_raw, train_end, te_idx, seq, N)
        pos_i = np.where(cl == 1)[0]; neg_i = np.where(cl == 0)[0]
        print(f"  concept pos={len(pos_i)} neg={len(neg_i)}")

        if len(pos_i) >= 3 and len(neg_i) >= 3:
            tcav = TemporalTCAV(pooling=pooling, cv_folds=tcav_folds, seed=seed)
            tcav.fit(Ea[pos_i], Ea[neg_i],
                     cam_a[pos_i] if pooling != "mean" else None,
                     cam_a[neg_i] if pooling != "mean" else None)
            cv = tcav.get_cv_df()
            cv.to_csv(os.path.join(pd_b, "tcav_cv.csv"), index=False)

            # H5
            r = stats.accuracy_above_chance(cv["accuracy"].values, 0.5, n_perm, seed)
            test_rows.append(dict(hyp="H5", metric="tcav_acc",
                                  stat=r["mean_acc"], p=r["p_value"],
                                  es=r["mean_acc"] - 0.5, n=r["n_folds"]))

            # H8
            r8 = stats.cosine_stability(tcav.fold_cavs_)
            test_rows.append(dict(hyp="H8", metric="cav_stability",
                                  stat=r8["mean_cos"], p=np.nan, es=r8["min_cos"],
                                  n=r8["n_pairs"],
                                  detail=f"mean={r8['mean_cos']:.3f} min={r8['min_cos']:.3f}"))

            # directional derivative + scores
            v_c = tcav.get_cav()
            dd = tcav.directional_derivative(dYdE)
            scores = tcav.aggregate(dd, cam_a)
            pri = scores["cam_weighted_pos_mass"]
            pd.DataFrame({"date": dates, "is_event": is_ev, "concept": cl,
                           **scores}).to_csv(os.path.join(pd_b, "scores.csv"), index=False)

            # H6
            ev_s, ne_s = pri[is_ev == 1], pri[is_ev == 0]
            if len(ev_s) > 2 and len(ne_s) > 2:
                r = stats.two_sample_perm(ev_s, ne_s, n_perm, seed)
                test_rows.append(dict(hyp="H6", metric="cwcr_ev_calm",
                                      stat=r["mean_diff"], p=r["p_value"],
                                      es=r["effect_size"], n=r["n_a"] + r["n_b"]))
            # H6b
            co_s, nc_s = pri[cl == 1], pri[cl == 0]
            if len(co_s) > 2 and len(nc_s) > 2:
                r = stats.two_sample_perm(co_s, nc_s, n_perm, seed)
                test_rows.append(dict(hyp="H6b", metric="cwcr_con_off",
                                      stat=r["mean_diff"], p=r["p_value"],
                                      es=r["effect_size"], n=r["n_a"] + r["n_b"]))

            # H7
            r7 = stats.interaction_perm(pri, is_ev, cl, n_perm, seed)
            test_rows.append(dict(hyp="H7", metric="cwcr_interaction",
                                  stat=r7["interaction"], p=r7["p_value"],
                                  es=r7["effect_size"], n=N))

            # ── plots ─────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 4))
            for idx, lbl, c in [(ev_i, "event", "tab:red"), (ct_i, "calm", "tab:blue")]:
                if len(idx) == 0: continue
                m = dd[idx].mean(0); se = dd[idx].std(0) / np.sqrt(len(idx))
                ax.plot(m, label=lbl, color=c)
                ax.fill_between(range(len(m)), m - se, m + se, alpha=.15, color=c)
            ax.axhline(0, color="gray", ls="--", lw=.5)
            ax.set_xlabel("Timestep"); ax.set_ylabel("dd_t")
            ax.set_title("Temporal Directional Derivative"); ax.legend(); ax.grid(alpha=.2)
            plt.tight_layout(); _sfig(fig, os.path.join(pd_b, "dd_timecourse.png"), show)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot([ne_s, ev_s], labels=["calm", "event"])
            ax.set_title("CWCR: Event vs Calm"); ax.grid(alpha=.2, axis="y")
            plt.tight_layout(); _sfig(fig, os.path.join(pd_b, "cwcr_boxplot.png"), show)

            if len(ev_i) > 0:
                ex = ev_i[0]; T = len(raw_w[ex])
                fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                axes[0].plot(raw_w[ex], color="k"); axes[0].set_ylabel(target_col)
                axes[0].set_title(f"Example (idx={ex})")
                axes[1].fill_between(range(T), 0, cam_a[ex], alpha=.6, color="steelblue")
                axes[1].set_ylabel("|CAM|")
                axes[2].plot(dd[ex], color="purple"); axes[2].axhline(0, c="gray", ls="--")
                axes[2].set_ylabel("dd_t")
                cw = np.clip(cam_a[ex], 0, None) * np.clip(dd[ex], 0, None)
                axes[3].fill_between(range(T), 0, cw, alpha=.6, color="tab:orange")
                axes[3].set_ylabel("CAM×max(dd,0)"); axes[3].set_xlabel("t")
                for a in axes: a.grid(alpha=.2)
                plt.tight_layout(); _sfig(fig, os.path.join(pd_b, "example.png"), show)

            tcav_res = dict(tcav=tcav, dd=dd, scores=scores, v_c=v_c, cl=cl)
        else:
            print("[Analysis] Not enough concept samples")

    # ══════════════════════════════════════════════════════════
    # FDR + Summary
    # ══════════════════════════════════════════════════════════
    tdf = pd.DataFrame(test_rows)
    if len(tdf) > 0:
        valid = tdf["p"].notna()
        if valid.sum() > 0:
            tdf.loc[valid, "p_fdr"] = stats.benjamini_hochberg(tdf.loc[valid, "p"].values)
            tdf["sig"] = tdf["p_fdr"] < 0.05
    tdf.to_csv(os.path.join(save_dir, "all_tests.csv"), index=False)

    sig = int(tdf["sig"].sum()) if "sig" in tdf.columns else 0
    summary = dict(n_tests=len(tdf), n_sig=sig, target=target_idx)
    json.dump(summary, open(os.path.join(save_dir, "summary.json"), "w"), indent=2)

    print(f"\n{'='*60}")
    print(f" RESULTS: {sig}/{len(tdf)} significant (FDR<0.05)")
    print(f"{'='*60}")
    for _, r in tdf.iterrows():
        star = "✓" if r.get("sig") else " "
        ps = f"{r['p']:.4f}" if pd.notna(r["p"]) else "N/A"
        print(f" [{star}] {r['hyp']:4s} {r['metric']:25s} "
              f"stat={r['stat']:+.4f} p={ps} es={r.get('es',0):+.3f}")

    return dict(test_df=tdf, dtw_df=dtw_df, cam_s=cam_s, cam_a=cam_a,
                Ea=Ea, tcav=tcav_res, is_ev=is_ev, dates=dates, N=N,
                save_dir=save_dir, summary=summary)