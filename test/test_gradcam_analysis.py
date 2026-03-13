"""
Tests for gradcam.py, tcav_temporal.py, stats.py, analysis.py.

Usage:
    python test_gradcam_analysis.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from vix_xai.config import Config, set_seed
from vix_xai.data import load_frame, build_dataloaders
from vix_xai.models import TCNEnsemble, count_parameters
from vix_xai.training import train_model
from vix_xai.xai import collect_test_windows


# ═══════════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════════

def _make_csv(path, n=500):
    np.random.seed(0)
    df = pd.DataFrame({
        "날짜": pd.bdate_range("2020-01-01", periods=n),
        "SPX": 3000 + np.cumsum(np.random.randn(n) * 10),
        "VIX": np.clip(15 + np.cumsum(np.random.randn(n) * 0.5), 8, 80),
        "Gold": 1500 + np.cumsum(np.random.randn(n) * 5),
    })
    df.to_csv(path, index=False)
    return path


def _setup():
    """작은 모델을 학습시키고 필요한 객체를 반환."""
    set_seed(42)
    csv = _make_csv("/tmp/test_analysis.csv")
    device = torch.device("cpu")
    cfg = Config(
        csv_path=csv, index_col="날짜", drop_cols=(), target_col="VIX",
        seq_len=10, batch_size=32, epochs=3, patience=3, min_epoch=1,
        tcn_channels=(2, 2), tcn_kernel=3, fc_hidden=(8,),
        param_budget=50000, use_amp=False, num_workers=0,
    )
    df_raw = load_frame(cfg.csv_path, cfg.index_col)
    dl_tr, dl_va, dl_te, meta = build_dataloaders(
        df_raw=df_raw, target_col=cfg.target_col,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size,
        train_ratio=0.7, val_ratio=0.15,
        num_workers=0, pin_memory=False, persistent_workers=False,
        target_mode="level",
    )
    model = TCNEnsemble(len(meta["feature_names"]), meta["target_index"], cfg).to(device)
    model, _ = train_model(model, dl_tr, dl_va, cfg, device, nn.MSELoss(), "test")
    return model, meta, cfg, df_raw, dl_te, device


# ═══════════════════════════════════════════════════════════════════
# stats.py tests
# ═══════════════════════════════════════════════════════════════════

def test_stats_two_sample_perm():
    from vix_xai.stats import two_sample_perm
    a = np.random.randn(50) + 1.0  # shifted
    b = np.random.randn(50)
    r = two_sample_perm(a, b, n_perm=500, seed=42)
    assert r["mean_diff"] > 0
    assert 0 <= r["p_value"] <= 1
    assert "n_a" in r and "n_b" in r
    print(f"  two_sample_perm: diff={r['mean_diff']:.3f} p={r['p_value']:.3f}")


def test_stats_paired_perm():
    from vix_xai.stats import paired_perm
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    r = paired_perm(a, b, n_perm=500, seed=42)
    assert r["mean_diff"] > 0
    assert "n" in r


def test_stats_alignment():
    from vix_xai.stats import alignment_perm
    np.random.seed(0)
    cams = np.random.rand(20, 10)
    raw = np.random.rand(20, 10)
    r = alignment_perm(cams, raw, n_perm=200, seed=42)
    assert "real_mean" in r
    assert "p_value" in r
    print(f"  alignment: real={r['real_mean']:.3f} null={r['null_mean']:.3f}")


def test_stats_accuracy_above_chance():
    from vix_xai.stats import accuracy_above_chance
    # 높은 accuracy → significant
    r = accuracy_above_chance(np.array([0.8, 0.85, 0.9, 0.75, 0.82]), n_perm=500)
    assert r["p_value"] < 0.1
    # 낮은 accuracy → not significant
    r2 = accuracy_above_chance(np.array([0.5, 0.48, 0.52, 0.49, 0.51]), n_perm=500)
    assert r2["p_value"] > 0.1


def test_stats_interaction():
    from vix_xai.stats import interaction_perm
    np.random.seed(0)
    # 강한 interaction: event+concept에서만 높음
    n = 100
    ev = np.array([1] * 25 + [0] * 75)
    co = np.array(([1] * 12 + [0] * 13) + ([1] * 37 + [0] * 38))
    scores = np.random.randn(n)
    scores[(ev == 1) & (co == 1)] += 2.0  # interaction
    r = interaction_perm(scores, ev, co, n_perm=500, seed=42)
    assert "interaction" in r
    print(f"  interaction: {r['interaction']:.3f} p={r['p_value']:.3f}")


def test_stats_cosine():
    from vix_xai.stats import cosine_stability
    cavs = [np.array([1, 0, 0], dtype=float),
            np.array([0.99, 0.1, 0], dtype=float) / np.linalg.norm([0.99, 0.1, 0]),
            np.array([0.95, 0.2, 0.1], dtype=float) / np.linalg.norm([0.95, 0.2, 0.1])]
    r = cosine_stability(cavs)
    assert r["mean_cos"] > 0.8
    print(f"  cosine: mean={r['mean_cos']:.3f} min={r['min_cos']:.3f}")


def test_stats_bootstrap():
    from vix_xai.stats import block_bootstrap_ci
    v = np.random.randn(100) + 1.0
    r = block_bootstrap_ci(v, block_len=10, n_boot=500)
    assert r["ci_low"] < r["mean"] < r["ci_high"]


def test_stats_fdr():
    from vix_xai.stats import benjamini_hochberg
    pvals = np.array([0.01, 0.04, 0.05, 0.2, 0.5])
    corrected = benjamini_hochberg(pvals)
    assert np.all(corrected >= pvals)
    assert np.all(corrected <= 1.0)


# ═══════════════════════════════════════════════════════════════════
# gradcam.py tests
# ═══════════════════════════════════════════════════════════════════

def test_gradcam_branch():
    model, meta, cfg, df_raw, dl_te, device = _setup()
    from vix_xai.gradcam import TimeSeriesGradCAM, resolve_target_branch
    target = resolve_target_branch(model, meta, cfg)
    eng = TimeSeriesGradCAM(model, device)
    X_te = collect_test_windows(dl_te)
    cam_s, cam_a = eng.generate(X_te[0:1], target)
    eng.cleanup()
    assert cam_s.shape == (cfg.seq_len,), f"Expected ({cfg.seq_len},), got {cam_s.shape}"
    assert cam_a.shape == (cfg.seq_len,)
    assert cam_a.min() >= 0
    print(f"  branch CAM OK shape={cam_s.shape}")


def test_gradcam_batch():
    model, meta, cfg, _, dl_te, device = _setup()
    from vix_xai.gradcam import TimeSeriesGradCAM, resolve_target_branch
    target = resolve_target_branch(model, meta, cfg)
    eng = TimeSeriesGradCAM(model, device)
    X_te = collect_test_windows(dl_te)[:5]  # 5 samples
    cs, ca = eng.generate_batch(X_te, target, desc="test")
    eng.cleanup()
    assert cs.shape == (5, cfg.seq_len)
    print(f"  batch CAM OK shape={cs.shape}")


def test_temporal_gradient():
    model, meta, cfg, _, dl_te, device = _setup()
    from vix_xai.gradcam import TemporalGradientExtractor, resolve_target_branch
    target = resolve_target_branch(model, meta, cfg)
    gex = TemporalGradientExtractor(model, device, target)
    X_te = collect_test_windows(dl_te)
    E, G = gex.extract_single(X_te[0:1])
    gex.cleanup()
    assert E.ndim == 2  # (T, C)
    assert G.ndim == 2  # (T, C)
    assert E.shape == G.shape
    print(f"  gradient OK E={E.shape} G={G.shape}")


def test_embeddings():
    model, meta, cfg, _, dl_te, device = _setup()
    from vix_xai.gradcam import extract_embeddings, resolve_target_branch
    target = resolve_target_branch(model, meta, cfg)
    X_te = collect_test_windows(dl_te)[:5]
    Ea = extract_embeddings(model, X_te, target)
    assert Ea.ndim == 3  # (N, T, C)
    assert Ea.shape[0] == 5
    print(f"  embeddings OK shape={Ea.shape}")


# ═══════════════════════════════════════════════════════════════════
# tcav_temporal.py tests
# ═══════════════════════════════════════════════════════════════════

def test_tcav_mean_pooling():
    from vix_xai.tcav_temporal import TemporalTCAV
    np.random.seed(0)
    E_pos = np.random.randn(30, 10, 4) + 0.5
    E_neg = np.random.randn(30, 10, 4) - 0.5
    tcav = TemporalTCAV(pooling="mean", cv_folds=3, seed=42)
    tcav.fit(E_pos, E_neg)
    cv = tcav.get_cv_df()
    assert len(cv) == 3
    assert cv["accuracy"].mean() > 0.5
    v = tcav.get_cav()
    assert v.shape == (4,)
    print(f"  TCAV mean: acc={cv['accuracy'].mean():.3f}")


def test_tcav_cam_weighted():
    from vix_xai.tcav_temporal import TemporalTCAV
    np.random.seed(0)
    E_pos = np.random.randn(30, 10, 4) + 0.5
    E_neg = np.random.randn(30, 10, 4) - 0.5
    cam_pos = np.abs(np.random.randn(30, 10))
    cam_neg = np.abs(np.random.randn(30, 10))
    tcav = TemporalTCAV(pooling="cam_weighted", cv_folds=3, seed=42)
    tcav.fit(E_pos, E_neg, cam_pos, cam_neg)
    v = tcav.get_cav()
    assert v.shape == (4,)
    print(f"  TCAV cam_weighted: acc={tcav.get_cv_df()['accuracy'].mean():.3f}")


def test_tcav_directional_derivative():
    from vix_xai.tcav_temporal import TemporalTCAV
    np.random.seed(0)
    E_pos = np.random.randn(20, 10, 4) + 0.5
    E_neg = np.random.randn(20, 10, 4) - 0.5
    tcav = TemporalTCAV(pooling="mean", cv_folds=3, seed=42)
    tcav.fit(E_pos, E_neg)
    dYdE = np.random.randn(5, 10, 4)
    dd = tcav.directional_derivative(dYdE)
    assert dd.shape == (5, 10)
    scores = tcav.aggregate(dd, cam=np.abs(np.random.randn(5, 10)))
    assert "cam_weighted_pos_mass" in scores
    assert scores["cam_weighted_pos_mass"].shape == (5,)
    print(f"  dd OK shape={dd.shape}, scores keys={list(scores.keys())}")


# ═══════════════════════════════════════════════════════════════════
# Integration: full pipeline
# ═══════════════════════════════════════════════════════════════════

def test_full_pipeline():
    """run_analysis smoke test with tiny data."""
    model, meta, cfg, df_raw, dl_te, device = _setup()
    from vix_xai.analysis import run_analysis
    result = run_analysis(
        model=model, meta=meta, cfg=cfg,
        df_raw=df_raw, dl_te=dl_te, device=device,
        horizon=3, q_event=0.90, min_gap=10,
        n_perm=100, n_boot=100,  # small for speed
        save_dir="/tmp/test_analysis_output",
        seed=42,
    )
    if result is None:
        print("  full pipeline: not enough events (OK for tiny data)")
        return
    assert "test_df" in result
    assert "summary" in result
    tdf = result["test_df"]
    print(f"  full pipeline: {len(tdf)} tests, {tdf.get('sig', pd.Series()).sum()} sig")
    print("  PASS")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    passed = failed = 0
    tests = sorted(
        [(name, fn) for name, fn in globals().items()
         if name.startswith("test_") and callable(fn)],
        key=lambda x: x[0],
    )
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"{passed} passed, {failed} failed")
