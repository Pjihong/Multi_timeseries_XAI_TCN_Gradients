# VIX-XAI: Explainable VIX Prediction with TimeSeriesGradCAM and Temporal TCAV

Per-feature branch TCN/CNN ensemble for VIX forecasting, with a rigorous GradCAM-centered explainability framework validated through 8 statistical hypotheses.

---

## Overview

This project consists of two layers:

1. **Prediction** — A per-feature branch TCN or CNN ensemble with RevIN normalization that forecasts VIX levels (or log-VIX).
2. **Explanation** — A GradCAM-centered analysis pipeline that statistically tests whether the model attends differently to VIX spike events versus calm periods, using both DTW-based and TCAV-based validation.

The explanation pipeline is designed to answer:

- **When** does the model focus? → Target-branch GradCAM
- **What direction** is it sensitive to at those moments? → Temporal TCAV directional derivative
- **Is the difference statistically significant?** → Permutation tests with FDR correction

---

## Repository Structure

```
.
├── src/vix_xai/                    # Core package
│   │
│   │  ── Prediction ──────────────────────────────────
│   ├── config.py                   # Config dataclass, seed control, device
│   ├── data.py                     # Data loading, splitting, SequenceDataset
│   ├── models.py                   # RevIN, SingleTCN, SingleCNN, TCN/CNN Ensemble
│   ├── training.py                 # Training loop, early stopping
│   ├── eval.py                     # Level-RMSE evaluation, baselines
│   ├── experiments.py              # CNN architecture search, experiment suite
│   ├── utils.py                    # Plotting, model bundle save/load
│   │
│   │  ── Explainability (legacy) ─────────────────────
│   ├── xai.py                      # TimeSeriesGradCAMRegression (aggregate CAM)
│   ├── event_wraping.py            # DTW, distributional DTW, event weighting
│   ├── posthoc.py                  # Matched-pair analysis, deletion test
│   ├── metrics.py                  # Reference-based DTW metrics over time
│   ├── concepts.py                 # TCAVExtractorCV, C-DEW analysis
│   │
│   │  ── Explainability (new) ────────────────────────
│   ├── gradcam.py                  # Target-branch GradCAM + TemporalGradientExtractor
│   ├── tcav_temporal.py            # TemporalTCAV (CAV + dd_t + scoring)
│   ├── stats.py                    # 8 hypothesis tests + FDR correction
│   ├── analysis.py                 # run_analysis() unified pipeline
│   │
│   └── __init__.py
│
├── output/                         # Entry-point wrapper scripts
│   ├── vix_tcn_revin_xai_plus.py   # Main training + legacy XAI
│   ├── posthoc_analysis_v2.py      # Legacy post-hoc analysis
│   ├── cdew_concepts_v2.py         # Legacy C-DEW concept analysis
│   ├── metrics_over_time_v2.py     # Legacy DTW metrics
│   ├── event_warping.py            # Legacy event warping
│   ├── gradcam_tcav_analysis.py    # New: GradCAM + TCAV analysis wrapper
│   └── run_analysis.py             # New: Full analysis notebook-style script
│
├── tests/
│   ├── test_smoke_train.py         # Smoke test: train, evaluate, Grad-CAM
│   ├── test_event_wraping.py       # DTW module tests
│   └── test_gradcam_analysis.py    # New: All 4 new modules (15 tests)
│
├── outputs/                        # Generated results (gitignored)
│   ├── bundles/                    # Trained model bundles (.pt)
│   ├── experiments/                # Experiment suite results
│   ├── analysis/                   # GradCAM + TCAV analysis results
│   └── event_target_sweep/         # Parameter sweep results
│
├── timeseries_data.csv             # Input data (not included)
└── README.md
```

---

## Model Architecture

Per-feature branch ensemble with RevIN:

```
Feature 0 (log_VIX) ──→ [CNN/TCN branch 0] ──→ embed_0 ──┐
Feature 1 (SPX)     ──→ [CNN/TCN branch 1] ──→ embed_1 ──┤
Feature 2 (Gold)    ──→ [CNN/TCN branch 2] ──→ embed_2 ──┼──→ [FC head] ──→ ŷ
...                 ──→ [CNN/TCN branch F] ──→ embed_F ──┘
```

Each branch independently processes one feature through stacked Conv1d layers. RevIN provides instance normalization with learnable affine parameters. The FC head concatenates all branch outputs for the final prediction.

**Key detail**: When `target_mode="log"`, the target column becomes `log_VIX` (index 0), while the original `VIX` remains as a separate feature (index 4). All XAI analysis must use the model's actual target branch (index 0), not the raw VIX branch.

---

## Explainability Framework

### Why Target-Branch CAM (not Aggregate)

The legacy `xai.py` averages CAM across all branches. This loses the answer to "where does the model look **in the VIX series**?" because it mixes attention from SPX, Gold, etc.

The new `gradcam.py` extracts CAM from a single target branch:

```python
# Legacy (wrong for targeted analysis)
cam_agg, per_branch = legacy_cam.generate(x)  # mean of all branches

# New (correct)
cam_s, cam_a = target_cam.generate(x, branch_idx=0)  # target branch only
```

### GradCAM + TCAV Connection

```
GradCAM:    cam_t           → "timestep t is important" (scalar per t)
Gradient:   dY/dE_t         → "sensitive in this direction at t" (vector per t)  
TCAV:       v_c             → "concept direction" (vector)
Combination: dd_t = ⟨dY/dE_t, v_c⟩  → "concept sensitivity at t" (scalar per t)

Primary metric:
  CWCR = Σ_t cam̃_t × max(dd_t, 0)   → "concept sensitivity at important timesteps"
```

### Statistical Hypotheses

| ID | Hypothesis | Test Method | Function |
|----|-----------|-------------|----------|
| **H1** | CAM-DTW distance differs: event ≠ calm | Two-sample permutation (label shuffle) | `stats.two_sample_perm` |
| **H2** | AUC(CAM-DTW) > AUC(Raw DTW) | Paired bootstrap on same samples | `stats.paired_bootstrap_auc` |
| **H3** | CAM peaks align with market moves > chance | Within-window CAM shuffle | `stats.alignment_perm` |
| **H4** | CAM-important deletion Δ > random deletion Δ | Paired sign-flip permutation | `stats.paired_perm` + `stats.deletion_test` |
| **H5** | TCAV classification accuracy > 0.5 | Permutation against chance | `stats.accuracy_above_chance` |
| **H6** | CWCR differs: event ≠ calm | Two-sample permutation | `stats.two_sample_perm` |
| **H6b** | CWCR differs: concept-on ≠ concept-off | Two-sample permutation | `stats.two_sample_perm` |
| **H7** | Event × concept interaction on CWCR | 2×2 interaction contrast permutation | `stats.interaction_perm` |
| **H8** | CAV stable across CV folds | Pairwise cosine similarity | `stats.cosine_stability` |

All p-values undergo Benjamini-Hochberg FDR correction. Non-overlap subsampling (`min_gap ≥ seq_len`) prevents inflated significance from sliding-window dependency.

---

## Quick Start

### 1. Install Dependencies

```bash
conda create -n vixenv python=3.10
conda activate vixenv
pip install torch numpy pandas scikit-learn scipy matplotlib tqdm
```

### 2. Train a Model

```bash
python output/vix_tcn_revin_xai_plus.py
# → saves outputs/bundles/best_model_bundle.pt
```

### 3. Run Tests

```bash
python tests/test_smoke_train.py
python tests/test_event_wraping.py
python tests/test_gradcam_analysis.py
# Expected: 15 passed, 0 failed (for test_gradcam_analysis)
```

### 4. Run GradCAM + TCAV Analysis

```bash
python output/run_analysis.py
```

Or in Python:

```python
from vix_xai.analysis import run_analysis
from vix_xai.utils import load_model_bundle
from vix_xai.data import load_frame, build_dataloaders
from vix_xai.config import Config, get_device

device = get_device()
model, meta_l, snap = load_model_bundle('outputs/bundles/best_model_bundle.pt', device)
cfg = Config(**snap['cfg'])
df_raw = load_frame(cfg.csv_path, cfg.index_col, list(cfg.drop_cols))

_, _, dl_te, meta = build_dataloaders(
    df_raw=df_raw, target_col=cfg.target_col,
    seq_len=cfg.seq_len, batch_size=cfg.batch_size,
    train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio,
    num_workers=0, pin_memory=False, persistent_workers=False,
    target_mode=snap['target_mode'],
)

result = run_analysis(
    model=model, meta=meta, cfg=cfg,
    df_raw=df_raw, dl_te=dl_te, device=device,
    target_branch=int(meta["target_index"]),  # must be 0 for log mode
    save_dir="outputs/analysis",
)

print(result["test_df"])  # H1–H8 results with FDR correction
```

### 5. Outputs

```
outputs/analysis/
├── config.json                 # Run configuration
├── events.csv                  # Event/control labels per window
├── all_tests.csv               # All H1–H8 results + FDR
├── summary.json                # Counts of significant tests
├── mean_cam.png                # Event vs calm target-branch CAM
├── path_a/
│   ├── dtw.csv                 # DTW distances per window
│   └── boxplots.png            # DTW distribution comparison
└── path_b/
    ├── tcav_cv.csv             # TCAV cross-validation scores
    ├── scores.csv              # CWCR and secondary scores
    ├── dd_timecourse.png       # Temporal directional derivative curves
    ├── cwcr_boxplot.png        # CWCR event vs calm
    └── example.png             # Single-window decomposition
```

---

## Technical Notes

### In-Place Operation Safety

PyTorch `LeakyReLU(inplace=True)` conflicts with `register_full_backward_hook`. The solution in `gradcam.py`:

```python
# Forward hook returns clone → breaks view chain → in-place safe downstream
def fwd_hook(m, inp, out):
    cloned = out.clone()
    cloned.register_hook(lambda g: ...)  # tensor-level gradient capture
    return cloned  # replaces module output
```

### Target Branch Resolution

When `target_mode="log"`, features become `['log_VIX', ..., 'VIX', ...]`. Using `cfg.target_col="VIX"` would select the wrong branch (raw VIX, not the model's actual target). Always use:

```python
target_branch = int(meta["target_index"])  # from build_dataloaders
```

### Concept Definition

The default concept is "Flight-to-Safety" (Gold return above 90th percentile of training period AND equity index declining). Custom concepts can be passed via the `concept_definitions` parameter. Threshold is computed from training data only to prevent leakage.

---

## File Descriptions

### Core Package (`src/vix_xai/`)

| File | Lines | Description |
|------|-------|-------------|
| `config.py` | ~80 | `Config` dataclass, `set_seed()`, `get_device()` |
| `data.py` | ~170 | `load_frame()`, `split_by_time()`, `SequenceDataset`, `build_dataloaders()` |
| `models.py` | ~260 | `RevIN`, `SingleTCN`, `SingleCNN`, `TCNEnsemble`, `CNNEnsemble` |
| `training.py` | ~100 | `train_model()` with AMP, gradient clipping, early stopping |
| `eval.py` | ~80 | `evaluate_level_rmse()`, `compute_baselines()` |
| `xai.py` | ~230 | `TimeSeriesGradCAMRegression` (aggregate CAM, legacy) |
| `event_wraping.py` | ~450 | DTW, distributional DTW, event weighting, cost matrices |
| `posthoc.py` | ~550 | Matched-pair analysis, deletion test, `GradCAMEngine` (legacy) |
| `metrics.py` | ~500 | Reference-based DTW metrics, AUC evaluation |
| `concepts.py` | ~500 | `TCAVExtractorCV`, C-DEW analysis, concept dashboard |
| `utils.py` | ~150 | Plot helpers, model bundle save/load |
| `experiments.py` | ~350 | CNN architecture search, `run_experiment_suite()` |
| **`gradcam.py`** | ~270 | `TimeSeriesGradCAM` (target-branch), `TemporalGradientExtractor` |
| **`tcav_temporal.py`** | ~195 | `TemporalTCAV` (CAV + dd_t + CWCR scoring) |
| **`stats.py`** | ~336 | 8 hypothesis tests, BH FDR, block bootstrap, deletion test |
| **`analysis.py`** | ~428 | `run_analysis()` unified pipeline (Path A + Path B) |

### Wrapper Scripts (`output/`)

| File | Purpose |
|------|---------|
| `vix_tcn_revin_xai_plus.py` | Training + legacy XAI entry point |
| `posthoc_analysis_v2.py` | Legacy post-hoc matched-pair analysis |
| `cdew_concepts_v2.py` | Legacy C-DEW concept analysis |
| `metrics_over_time_v2.py` | Legacy reference-based DTW metrics |
| `event_warping.py` | Legacy event warping wrapper |
| **`gradcam_tcav_analysis.py`** | New GradCAM + TCAV wrapper |
| **`run_analysis.py`** | New full analysis script (notebook-style) |

### Tests (`tests/`)

| File | Tests | Description |
|------|-------|-------------|
| `test_smoke_train.py` | 1 | Build model, train, evaluate, Grad-CAM |
| `test_event_wraping.py` | 9 | DTW distance functions, event weighting |
| **`test_gradcam_analysis.py`** | 15 | All 4 new modules: stats, gradcam, tcav, pipeline |

---

## Citation

If you use this framework, please cite:

```bibtex
@misc{vix_xai_2025,
  title={Explainable VIX Prediction with TimeSeriesGradCAM and Temporal TCAV},
  year={2025},
}
```

---

## License

MIT
