# %% [markdown]
# # GradCAM + Temporal TCAV Analysis
#
# JupyterLab에서 이 파일을 열고 셀 단위로 실행하세요.
# (Open as Notebook 또는 그냥 .py로 실행)

# %% [markdown]
# ## 0. 환경 설정

# %%
import sys
from pathlib import Path

# ─── 프로젝트 경로 설정 ─────────────────────────────────────
# 본인의 프로젝트 루트 경로로 수정하세요
PROJECT_ROOT = Path(".").resolve()  # 현재 디렉토리가 프로젝트 루트일 때
# PROJECT_ROOT = Path("/home/jihong/your_project")  # 또는 직접 지정

SRC_DIR = PROJECT_ROOT / "src"
assert SRC_DIR.exists(), f"src/ 디렉토리가 없습니다: {SRC_DIR}"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print(f"Project root: {PROJECT_ROOT}")
print(f"Source dir:   {SRC_DIR}")

# %%
# ─── 패키지 import 확인 ─────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# 기존 모듈
from vix_xai.config import Config, set_seed, get_device
from vix_xai.data import load_frame, build_dataloaders
from vix_xai.models import TCNEnsemble, CNNEnsemble, count_parameters
from vix_xai.training import train_model
from vix_xai.eval import evaluate_level_rmse
from vix_xai.xai import TimeSeriesGradCAMRegression, collect_test_windows
from vix_xai.utils import load_model_bundle

# 새 모듈
from vix_xai.gradcam import TimeSeriesGradCAM, TemporalGradientExtractor, extract_embeddings, resolve_target_branch
from vix_xai.tcav_temporal import TemporalTCAV
from vix_xai import stats
from vix_xai.analysis import run_analysis

print("모든 import 성공 ✓")
print(f"PyTorch: {torch.__version__}")
print(f"Device:  {get_device()}")

# %% [markdown]
# ## 1. 모델 로드
#
# 두 가지 방법:
# - **방법 A**: 이미 학습된 모델 bundle 로드
# - **방법 B**: 처음부터 학습

# %%
# ═══════════════════════════════════════════════════════════════
# 방법 A: 기존 학습된 모델 로드 (bundle이 있는 경우)
# ═══════════════════════════════════════════════════════════════

BUNDLE_PATH = PROJECT_ROOT / "outputs" / "bundles" / "best_model_bundle.pt"
CSV_PATH = PROJECT_ROOT / "timeseries_data.csv"  # 데이터 경로

# CSV 경로가 다르면 수정하세요
# CSV_PATH = PROJECT_ROOT / "data" / "your_data.csv"

device = get_device()
print(f"Device: {device}")

if BUNDLE_PATH.exists():
    print(f"Bundle 로드: {BUNDLE_PATH}")
    model, meta_loaded, snapshot = load_model_bundle(str(BUNDLE_PATH), device)
    model.eval()
    print(f"Model loaded: {type(model).__name__}")
    print(f"Arch: {snapshot['arch']}, target_mode: {snapshot['target_mode']}")

    # Config 복원
    cfg = Config(**snapshot['cfg']) if isinstance(snapshot['cfg'], dict) else snapshot['cfg']
else:
    print(f"Bundle 없음: {BUNDLE_PATH}")
    print("→ 방법 B (아래 셀)로 학습하세요")
    model, meta_loaded, cfg = None, None, None

# %%
# ═══════════════════════════════════════════════════════════════
# 방법 B: 처음부터 학습 (bundle이 없거나 새로 학습할 때)
# ═══════════════════════════════════════════════════════════════

if model is None:
    set_seed(42)

    # Config 설정 — 본인 데이터에 맞게 수정
    cfg = Config(
        csv_path=str(CSV_PATH),
        index_col="날짜",           # 날짜 컬럼명
        drop_cols=("Silver", "Copper", "USD/GBP", "USD/CNY", "USD/JPY", "USD/EUR", "USD/CAD"),
        target_col="VIX",
        seq_len=20,
        batch_size=64,
        epochs=300,
        patience=20,
        min_epoch=50,
        lr=1e-4,
        tcn_channels=(3, 3),
        tcn_kernel=3,
        fc_hidden=(16,),
        param_budget=4000,
    )

    df_raw = load_frame(cfg.csv_path, cfg.index_col, list(cfg.drop_cols))
    print(f"Data shape: {df_raw.shape}, columns: {list(df_raw.columns)}")

    dl_tr, dl_va, dl_te, meta = build_dataloaders(
        df_raw=df_raw, target_col=cfg.target_col,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and device.type == "cuda"),
        persistent_workers=cfg.persistent_workers,
        target_mode="level",
    )

    model = TCNEnsemble(
        len(meta["feature_names"]), meta["target_index"], cfg
    ).to(device)
    n_total, n_train = count_parameters(model)
    print(f"Parameters: {n_total} total, {n_train} trainable")

    import torch.nn as nn
    model, hist = train_model(
        model, dl_tr, dl_va, cfg, device, nn.MSELoss(), "main"
    )
    print(f"Best val loss: {hist['best_val_loss']:.4f}")

    rmse, _, _ = evaluate_level_rmse(
        model, dl_te, device, meta["scaler_y"],
        "level", meta["df_te"], cfg.target_col, cfg.seq_len,
    )
    print(f"Test RMSE: {rmse:.4f}")

# %%
# ═══════════════════════════════════════════════════════════════
# DataLoader 준비 (bundle 로드 시에도 필요)
# ═══════════════════════════════════════════════════════════════

df_raw = load_frame(cfg.csv_path, cfg.index_col,
                     list(cfg.drop_cols) if cfg.drop_cols else None)

# target_mode 결정: snapshot에서 가져오거나 default "level"
if 'snapshot' in dir() and snapshot is not None:
    target_mode = snapshot.get("target_mode", "level")
else:
    target_mode = "level"
print(f"target_mode: {target_mode}")

# meta에 df_tr, df_va, df_te가 있어야 함 → 항상 다시 build
if 'meta' not in dir() or meta is None or 'df_te' not in meta:
    dl_tr, dl_va, dl_te, meta = build_dataloaders(
        df_raw=df_raw, target_col=cfg.target_col,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size,
        train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
        num_workers=0, pin_memory=False, persistent_workers=False,
        target_mode=target_mode,
    )

print(f"Train: {len(meta['df_tr'])}, Val: {len(meta['df_va'])}, Test: {len(meta['df_te'])}")
print(f"Features: {meta['feature_names']}")
print(f"Target: {meta['model_target_col']} (idx={meta['target_index']})")
print(f"→ target_branch will be: {meta['target_index']}  "
      f"(feature: {meta['feature_names'][meta['target_index']]})")

# %% [markdown]
# ## 2. GradCAM + TCAV 분석 실행
#
# `run_analysis()` 한 번 호출로 전체 분석 완료.

# %%
# ═══════════════════════════════════════════════════════════════
# 분석 실행
# ═══════════════════════════════════════════════════════════════

SAVE_DIR = str(PROJECT_ROOT / "outputs" / "gradcam_tcav")

result = run_analysis(
    model=model,
    meta=meta,
    cfg=cfg,
    df_raw=df_raw,
    dl_te=dl_te,
    device=device,
    target_col=cfg.target_col,
    target_branch=int(meta["target_index"]),  # 반드시 모델의 target index 사용

    # Event 정의
    horizon=5,           # 미래 5일 내 spike
    q_event=0.95,        # 상위 5%가 event
    positive_only=True,  # positive spike만 (VIX 급등)
    min_gap=20,          # non-overlap subsampling

    # Path A: DTW
    run_path_a=True,
    band=5,
    alpha=1.5,
    top_p=0.2,

    # Path B: Temporal TCAV
    run_path_b=True,
    pooling="cam_weighted",  # primary pooling
    tcav_folds=5,

    # 통계
    n_perm=5000,    # permutation 횟수 (작은 데이터면 1000으로)
    n_boot=2000,    # bootstrap 횟수

    save_dir=SAVE_DIR,
    show=False,     # True면 plot을 inline으로 표시
    seed=42,
)

# %%
if result is None:
    print("분석 실패: event/control 샘플 부족")
    print("→ q_event를 낮추거나 (예: 0.90) min_gap을 줄여보세요")
else:
    print(f"\n저장 위치: {result['save_dir']}")

# %% [markdown]
# ## 3. 결과 확인

# %%
# ═══════════════════════════════════════════════════════════════
# 가설 검정 결과 테이블
# ═══════════════════════════════════════════════════════════════

if result is not None:
    tdf = result["test_df"]
    display_cols = ["hyp", "metric", "stat", "p", "es"]
    if "p_fdr" in tdf.columns:
        display_cols.append("p_fdr")
    if "sig" in tdf.columns:
        display_cols.append("sig")

    print("=" * 70)
    print(" 가설 검정 결과")
    print("=" * 70)
    with pd.option_context("display.max_colwidth", 40, "display.float_format", "{:.4f}".format):
        print(tdf[display_cols].to_string(index=False))

# %%
# Significant 결과만 필터링
if result is not None and "sig" in tdf.columns:
    sig_df = tdf[tdf["sig"] == True]
    print(f"\n유의한 결과 ({len(sig_df)}/{len(tdf)}):")
    if len(sig_df) > 0:
        print(sig_df[display_cols].to_string(index=False))
    else:
        print("  유의한 결과 없음 (FDR < 0.05)")

# %% [markdown]
# ## 4. 시각화

# %%
# ═══════════════════════════════════════════════════════════════
# 저장된 플롯 로드 + 표시
# ═══════════════════════════════════════════════════════════════

if result is not None:
    save_dir = Path(result["save_dir"])

    plot_files = [
        save_dir / "mean_cam.png",
        save_dir / "path_a" / "boxplots.png",
        save_dir / "path_b" / "dd_timecourse.png",
        save_dir / "path_b" / "cwcr_boxplot.png",
        save_dir / "path_b" / "example.png",
    ]

    for pf in plot_files:
        if pf.exists():
            print(f"\n{'─'*50}")
            print(f" {pf.name}")
            print(f"{'─'*50}")
            img = plt.imread(str(pf))
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(img)
            ax.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print(f"  (없음: {pf})")

# %% [markdown]
# ## 5. 상세 분석

# %%
# ═══════════════════════════════════════════════════════════════
# CAM 분석: event vs calm의 temporal attention 패턴
# ═══════════════════════════════════════════════════════════════

if result is not None:
    cam_a = result["cam_a"]  # (N, T) absolute CAM
    is_ev = result["is_ev"]

    ev_cam = cam_a[is_ev == 1]
    ne_cam = cam_a[is_ev == 0]
    T = cam_a.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Mean CAM
    ax = axes[0]
    ax.plot(ev_cam.mean(0), label=f"event (n={len(ev_cam)})", color="tab:red")
    ax.plot(ne_cam.mean(0), label=f"calm (n={len(ne_cam)})", color="tab:blue")
    ax.set_xlabel("Timestep"); ax.set_ylabel("|CAM|")
    ax.set_title("Mean Target Branch CAM"); ax.legend(); ax.grid(alpha=0.2)

    # 2. CAM peak 분포
    ax = axes[1]
    ev_peaks = [np.argmax(cam_a[i]) for i in range(len(cam_a)) if is_ev[i] == 1]
    ne_peaks = [np.argmax(cam_a[i]) for i in range(len(cam_a)) if is_ev[i] == 0]
    ax.hist(ev_peaks, bins=T, alpha=0.5, color="tab:red", label="event", density=True)
    ax.hist(ne_peaks, bins=T, alpha=0.5, color="tab:blue", label="calm", density=True)
    ax.set_xlabel("Peak Timestep"); ax.set_ylabel("Density")
    ax.set_title("CAM Peak Distribution"); ax.legend(); ax.grid(alpha=0.2)

    # 3. CAM 집중도 (entropy)
    ax = axes[2]
    from vix_xai.gradcam import TimeSeriesGradCAM  # just for the figure
    ev_ent = [-np.sum(c / (c.sum() + 1e-12) * np.log(c / (c.sum() + 1e-12) + 1e-12)) for c in ev_cam]
    ne_ent = [-np.sum(c / (c.sum() + 1e-12) * np.log(c / (c.sum() + 1e-12) + 1e-12)) for c in ne_cam]
    ax.boxplot([ne_ent, ev_ent], labels=["calm", "event"])
    ax.set_ylabel("Entropy (lower = more focused)")
    ax.set_title("CAM Entropy"); ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(str(Path(SAVE_DIR) / "cam_detailed.png"), dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ═══════════════════════════════════════════════════════════════
# TCAV 분석: temporal directional derivative 상세
# ═══════════════════════════════════════════════════════════════

if result is not None and result.get("tcav") is not None:
    tcav_res = result["tcav"]
    dd = tcav_res["dd"]       # (N, T) directional derivative
    scores = tcav_res["scores"]
    cl = tcav_res["cl"]       # concept labels

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. dd_t: event vs calm
    ax = axes[0, 0]
    for mask, lbl, c in [(is_ev == 1, "event", "tab:red"), (is_ev == 0, "calm", "tab:blue")]:
        if mask.sum() > 0:
            m = dd[mask].mean(0)
            se = dd[mask].std(0) / np.sqrt(mask.sum())
            ax.plot(m, label=lbl, color=c)
            ax.fill_between(range(T), m - se, m + se, alpha=0.15, color=c)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_title("dd_t = ⟨∂ŷ/∂E_t, v_c⟩"); ax.legend(); ax.grid(alpha=0.2)

    # 2. dd_t: concept-on vs concept-off
    ax = axes[0, 1]
    for mask, lbl, c in [(cl == 1, "concept-on", "tab:green"), (cl == 0, "concept-off", "tab:gray")]:
        if mask.sum() > 0:
            m = dd[mask].mean(0)
            ax.plot(m, label=f"{lbl} (n={mask.sum()})", color=c)
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_title("dd_t by Concept Status"); ax.legend(); ax.grid(alpha=0.2)

    # 3. CWCR distribution
    ax = axes[1, 0]
    pri = scores["cam_weighted_pos_mass"]
    ax.hist(pri[is_ev == 1], bins=30, alpha=0.5, color="tab:red", label="event", density=True)
    ax.hist(pri[is_ev == 0], bins=30, alpha=0.5, color="tab:blue", label="calm", density=True)
    ax.set_title("CWCR Distribution"); ax.set_xlabel("CAM-weighted Positive Mass")
    ax.legend(); ax.grid(alpha=0.2)

    # 4. Score comparison (all 4 scores)
    ax = axes[1, 1]
    score_names = list(scores.keys())
    ev_means = [scores[s][is_ev == 1].mean() for s in score_names]
    ne_means = [scores[s][is_ev == 0].mean() for s in score_names]
    x = np.arange(len(score_names))
    w = 0.35
    ax.bar(x - w/2, ev_means, w, label="event", color="tab:red", alpha=0.7)
    ax.bar(x + w/2, ne_means, w, label="calm", color="tab:blue", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(score_names, rotation=30, ha="right", fontsize=8)
    ax.set_title("Score Means: Event vs Calm"); ax.legend(); ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(str(Path(SAVE_DIR) / "tcav_detailed.png"), dpi=150, bbox_inches="tight")
    plt.show()

# %%
# ═══════════════════════════════════════════════════════════════
# 예시 윈도우 분해 (CAM + dd_t + CAM×dd_t)
# ═══════════════════════════════════════════════════════════════

if result is not None and result.get("tcav") is not None:
    # event 윈도우 중 CWCR이 가장 높은 것
    ev_mask = (is_ev == 1)
    if ev_mask.sum() > 0:
        pri = scores["cam_weighted_pos_mass"]
        ev_indices = np.where(ev_mask)[0]
        top_idx = ev_indices[np.argmax(pri[ev_indices])]

        raw_lv = np.nan_to_num(df_raw.reindex(meta["df_te"].index)[cfg.target_col].to_numpy())
        raw_w = raw_lv[top_idx : top_idx + cfg.seq_len]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        t = np.arange(cfg.seq_len)

        axes[0].plot(t, raw_w, color="black", lw=1.5)
        axes[0].set_ylabel(cfg.target_col); axes[0].set_title(f"Top Event Window (idx={top_idx})")

        axes[1].fill_between(t, 0, cam_a[top_idx], alpha=0.7, color="steelblue")
        axes[1].set_ylabel("Target Branch |CAM|")

        axes[2].plot(t, dd[top_idx], color="purple", lw=1.5)
        axes[2].axhline(0, color="gray", ls="--", lw=0.5)
        axes[2].set_ylabel("dd_t = ⟨∂ŷ/∂E_t, v_c⟩")

        cw = np.clip(cam_a[top_idx], 0, None) * np.clip(dd[top_idx], 0, None)
        axes[3].fill_between(t, 0, cw, where=cw > 0, alpha=0.7, color="tab:orange", label="positive")
        axes[3].set_ylabel("CAM × max(dd_t, 0)")
        axes[3].set_xlabel("Timestep"); axes[3].legend()

        for ax in axes:
            ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(str(Path(SAVE_DIR) / "top_event_decomposition.png"), dpi=150, bbox_inches="tight")
        plt.show()

        print(f"Window idx={top_idx}")
        print(f"  CWCR = {pri[top_idx]:.6f}")
        print(f"  CAM peak at t={np.argmax(cam_a[top_idx])}")
        print(f"  dd_t peak at t={np.argmax(dd[top_idx])}")

# %% [markdown]
# ## 6. 결과 요약 CSV 확인

# %%
if result is not None:
    save_dir = Path(result["save_dir"])

    files = [
        save_dir / "all_tests.csv",
        save_dir / "events.csv",
        save_dir / "config.json",
        save_dir / "summary.json",
        save_dir / "path_a" / "dtw.csv",
        save_dir / "path_b" / "tcav_cv.csv",
        save_dir / "path_b" / "scores.csv",
    ]

    for f in files:
        if f.exists():
            print(f"✓ {f.relative_to(save_dir)}")
            if f.suffix == ".csv":
                df = pd.read_csv(f)
                print(f"  shape: {df.shape}, columns: {list(df.columns)[:6]}...")
            elif f.suffix == ".json":
                import json
                print(f"  {json.load(open(f))}")
        else:
            print(f"✗ {f.relative_to(save_dir)}")
    print()

# %% [markdown]
# ## 7. Ablation: mean pooling vs cam_weighted pooling

# %%
if result is not None and result.get("tcav") is not None:
    print("Pooling 비교: mean vs cam_weighted")
    print("=" * 50)

    Ea = result["Ea"]
    cam_a = result["cam_a"]

    # 현재 결과 (cam_weighted)
    tcav_cw = result["tcav"]["tcav"]
    cv_cw = tcav_cw.get_cv_df()

    # mean pooling으로 재학습
    cl = result["tcav"]["cl"]
    pos_i, neg_i = np.where(cl == 1)[0], np.where(cl == 0)[0]

    if len(pos_i) >= 3 and len(neg_i) >= 3:
        tcav_mean = TemporalTCAV(pooling="mean", cv_folds=5, seed=42)
        tcav_mean.fit(Ea[pos_i], Ea[neg_i])
        cv_mean = tcav_mean.get_cv_df()

        # temporal gradient는 이미 추출됨
        gex = TemporalGradientExtractor(model, device, resolve_target_branch(model, meta, cfg))
        # 이미 추출된 dYdE가 없으면 다시 추출 필요
        # 여기서는 기존 tcav_res의 dd를 v_c만 바꿔서 재계산
        # (dYdE가 result에 없으므로 dd만 사용)

        print(f"  cam_weighted: CV acc = {cv_cw['accuracy'].mean():.3f} ± {cv_cw['accuracy'].std():.3f}")
        print(f"  mean:         CV acc = {cv_mean['accuracy'].mean():.3f} ± {cv_mean['accuracy'].std():.3f}")

        gex.cleanup()

# %% [markdown]
# ---
# ## 완료
#
# 결과 파일 위치: `outputs/gradcam_tcav/`
#
# 핵심 결과:
# - `all_tests.csv`: H1~H8 가설 검정 결과 (FDR 보정 포함)
# - `path_a/dtw.csv`: DTW 거리 전체
# - `path_b/scores.csv`: TCAV scores 전체
# - `mean_cam.png`: target branch CAM 비교
# - `path_b/example.png`: 분해 시각화