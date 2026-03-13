# VIX-XAI: TimeSeriesGradCAM과 Temporal TCAV를 활용한 설명 가능한 VIX 예측

특성별 분기(branch) TCN/CNN 앙상블을 사용하여 VIX를 예측하고, GradCAM 중심의 설명 가능성 프레임워크를 통해 이를 8개의 통계 가설로 검증하는 프로젝트입니다.

---

## 개요

이 프로젝트는 크게 두 계층으로 구성됩니다.

### 1) 예측(Prediction)

RevIN 정규화를 포함한 **특성별 분기 TCN 또는 CNN 앙상블**을 사용해 VIX 수준값(또는 log-VIX)을 예측합니다.

### 2) 설명(Explanation)

**GradCAM 중심 분석 파이프라인**을 통해, 모델이 VIX 급등 이벤트 구간과 안정 구간에서 서로 다른 방식으로 주의를 두는지 통계적으로 검정합니다. 이 과정에는 **DTW 기반 검증**과 **TCAV 기반 검증**이 포함됩니다.

이 설명 파이프라인은 다음 질문에 답하도록 설계되었습니다.

* **언제 모델이 집중하는가?** → Target-branch GradCAM
* **그 시점에서 어떤 방향성에 민감한가?** → Temporal TCAV directional derivative
* **그 차이가 통계적으로 유의한가?** → FDR 보정이 적용된 permutation test

---

## 저장소 구조

```text
.
├── src/vix_xai/                    # 핵심 패키지
│   │
│   │  ── Prediction ──────────────────────────────────
│   ├── config.py                   # Config dataclass, 시드 고정, 디바이스 설정
│   ├── data.py                     # 데이터 로딩, 분할, SequenceDataset
│   ├── models.py                   # RevIN, SingleTCN, SingleCNN, TCN/CNN 앙상블
│   ├── training.py                 # 학습 루프, early stopping
│   ├── eval.py                     # Level-RMSE 평가, 베이스라인
│   ├── experiments.py              # CNN 아키텍처 탐색, 실험 실행
│   ├── utils.py                    # 시각화, 모델 번들 저장/불러오기
│   │
│   │
│   │  ── Explainability (new) ────────────────────────
│   ├── gradcam.py                  # Target-branch GradCAM + TemporalGradientExtractor
│   ├── tcav_temporal.py            # TemporalTCAV (CAV + dd_t + scoring)
│   ├── stats.py                    # 8개 가설 검정 + FDR 보정
│   ├── analysis.py                 # run_analysis() 통합 파이프라인
│   │
│   └── __init__.py
│
├── output/                         # 실행용 래퍼 스크립트
│   ├── vix_tcn_revin_xai_plus.py   # 메인 학습 + legacy XAI
│   ├── gradcam_tcav_analysis.py    # GradCAM + TCAV 분석 래퍼
│   └── run_analysis.py             # 전체 분석용 notebook 스타일 스크립트
│
├── tests/
│   ├── test_smoke_train.py         # 스모크 테스트: 학습, 평가, Grad-CAM
│   ├── test_event_wraping.py       # DTW 모듈 테스트
│   └── test_gradcam_analysis.py    # 신규 4개 모듈 전체 테스트 (15개)
│
├── outputs/                        # 생성 결과 (gitignored)
│   ├── bundles/                    # 학습된 모델 번들 (.pt)
│   ├── experiments/                # 실험 결과
│   ├── analysis/                   # GradCAM + TCAV 분석 결과
│   └── event_target_sweep/         # 파라미터 스윕 결과
│
├── timeseries_data.csv             # 입력 데이터 (포함되지 않음)
└── README.md
```

---

## 모델 아키텍처

RevIN을 포함한 **특성별 분기 앙상블 구조**입니다.

```text
Feature 0 (log_VIX) ──→ [CNN/TCN branch 0] ──→ embed_0 ──┐
Feature 1 (SPX)     ──→ [CNN/TCN branch 1] ──→ embed_1 ──┤
Feature 2 (Gold)    ──→ [CNN/TCN branch 2] ──→ embed_2 ──┼──→ [FC head] ──→ ŷ
...                 ──→ [CNN/TCN branch F] ──→ embed_F ──┘
```

각 분기(branch)는 하나의 특성을 독립적으로 받아, 여러 개의 `Conv1d` 층을 거쳐 처리합니다. RevIN은 학습 가능한 affine 파라미터를 포함한 instance normalization을 제공합니다. 마지막으로 FC head가 각 분기의 출력을 이어 붙여 최종 예측값을 만듭니다.

### 핵심 주의사항

`target_mode="log"`일 때, 타깃 컬럼은 `log_VIX`(인덱스 0)가 되며, 원래의 `VIX`는 별도의 입력 특성(인덱스 4)으로 남습니다. 따라서 모든 XAI 분석은 **원본 VIX branch가 아니라, 모델의 실제 타깃 branch(인덱스 0)** 를 기준으로 수행해야 합니다.

---

## 설명 가능성 프레임워크

### 왜 Aggregate CAM이 아니라 Target-Branch CAM인가?

기존 `xai.py`는 모든 분기의 CAM을 평균냅니다. 하지만 이렇게 하면 SPX, Gold 등 다른 특성의 주의 정보까지 섞이기 때문에, **“모델이 VIX 시계열의 어디를 보고 있는가?”** 라는 질문에 정확히 답할 수 없습니다.

```python
# Legacy (targeted analysis에는 부적절)
cam_agg, per_branch = legacy_cam.generate(x)  # 모든 branch의 평균

# New (권장 방식)
cam_s, cam_a = target_cam.generate(x, branch_idx=0)  # 타깃 branch만 사용
```

### GradCAM과 TCAV의 연결

```text
GradCAM:    cam_t           → "시점 t가 중요하다" (시점별 스칼라)
Gradient:   dY/dE_t         → "시점 t에서 이 방향에 민감하다" (시점별 벡터)
TCAV:       v_c             → "개념 방향" (벡터)
조합:       dd_t = ⟨dY/dE_t, v_c⟩  → "시점 t에서 개념 민감도" (시점별 스칼라)

주요 지표:
  CWCR = Σ_t cam̃_t × max(dd_t, 0)
  → "중요한 시점에서의 개념 민감도"
```

---

## 통계 가설 검정

| ID      | 가설                                  | 검정 방법                                  | 함수                                          |
| ------- | ----------------------------------- | -------------------------------------- | ------------------------------------------- |
| **H1**  | CAM-DTW 거리는 이벤트 구간과 안정 구간에서 다르다     | Two-sample permutation (label shuffle) | `stats.two_sample_perm`                     |
| **H2**  | AUC(CAM-DTW) > AUC(Raw DTW)         | 동일 샘플 기반 paired bootstrap              | `stats.paired_bootstrap_auc`                |
| **H3**  | CAM peak는 우연보다 시장 움직임과 더 잘 정렬된다     | within-window CAM shuffle              | `stats.alignment_perm`                      |
| **H4**  | CAM 중요 구간 삭제에 따른 Δ가 랜덤 삭제보다 크다      | paired sign-flip permutation           | `stats.paired_perm` + `stats.deletion_test` |
| **H5**  | TCAV 분류 정확도 > 0.5                   | chance 대비 permutation test             | `stats.accuracy_above_chance`               |
| **H6**  | CWCR은 이벤트 구간과 안정 구간에서 다르다           | two-sample permutation                 | `stats.two_sample_perm`                     |
| **H6b** | CWCR은 concept-on과 concept-off에서 다르다 | two-sample permutation                 | `stats.two_sample_perm`                     |
| **H7**  | CWCR에 대해 Event × Concept 상호작용이 존재한다 | 2×2 interaction contrast permutation   | `stats.interaction_perm`                    |
| **H8**  | CAV는 CV fold 간 안정적이다                | pairwise cosine similarity             | `stats.cosine_stability`                    |

모든 p-value에는 **Benjamini-Hochberg FDR 보정**이 적용됩니다. 또한 `min_gap ≥ seq_len` 조건의 non-overlap subsampling을 사용해, sliding-window 의존성으로 인한 유의성 과대 추정을 방지합니다.

---

## 빠른 시작

### 1. 의존성 설치

```bash
conda create -n vixenv python=3.10
conda activate vixenv
pip install torch numpy pandas scikit-learn scipy matplotlib tqdm
```

### 2. 모델 학습

```bash
python output/vix_tcn_revin_xai_plus.py
# → outputs/bundles/best_model_bundle.pt 저장
```

### 3. 테스트 실행

```bash
python tests/test_smoke_train.py
python tests/test_event_wraping.py
python tests/test_gradcam_analysis.py
# 기대 결과: test_gradcam_analysis에서 15 passed, 0 failed
```

### 4. GradCAM + TCAV 분석 실행

```bash
python output/run_analysis.py
```

또는 Python 코드에서 직접 실행할 수 있습니다.

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
    target_branch=int(meta["target_index"]),  # log mode에서는 반드시 0
    save_dir="outputs/analysis",
)

print(result["test_df"])  # FDR 보정 포함 H1–H8 결과
```

### 5. 출력 결과

```text
outputs/analysis/
├── config.json                 # 실행 설정
├── events.csv                  # 윈도우별 event/control 라벨
├── all_tests.csv               # H1–H8 전체 결과 + FDR
├── summary.json                # 유의한 검정 개수 요약
├── mean_cam.png                # event vs calm target-branch CAM
├── path_a/
│   ├── dtw.csv                 # 윈도우별 DTW 거리
│   └── boxplots.png            # DTW 분포 비교
└── path_b/
    ├── tcav_cv.csv             # TCAV 교차검증 점수
    ├── scores.csv              # CWCR 및 보조 점수
    ├── dd_timecourse.png       # 시간축 directional derivative 곡선
    ├── cwcr_boxplot.png        # CWCR event vs calm 비교
    └── example.png             # 단일 윈도우 분해 예시
```

---

## 기술 메모

### In-Place 연산 안전성

PyTorch의 `LeakyReLU(inplace=True)`는 `register_full_backward_hook`와 충돌할 수 있습니다. `gradcam.py`에서는 다음 방식으로 이를 해결합니다.

```python
# Forward hook에서 clone을 반환해 view chain을 끊고,
# 이후 in-place 연산이 가능하도록 처리

def fwd_hook(m, inp, out):
    cloned = out.clone()
    cloned.register_hook(lambda g: ...)  # tensor-level gradient capture
    return cloned  # 모듈 출력을 교체
```

### Target Branch 선택

`target_mode="log"`이면 입력 특성은 `['log_VIX', ..., 'VIX', ...]` 형태가 됩니다. 이때 `cfg.target_col="VIX"`를 그대로 사용하면, 모델이 실제로 예측하는 branch가 아니라 **원본 VIX branch** 를 잘못 선택할 수 있습니다. 따라서 항상 아래처럼 지정해야 합니다.

```python
target_branch = int(meta["target_index"])  # build_dataloaders에서 가져오기
```

### 개념(Concept) 정의

기본 개념은 **“Flight-to-Safety”** 입니다.

* 금(Gold) 수익률이 학습 구간 기준 상위 90백분위 초과
* 동시에 주가지수가 하락

사용자 정의 개념은 `concept_definitions` 파라미터로 전달할 수 있습니다. 임곗값은 **데이터 누수 방지**를 위해 학습 데이터만으로 계산됩니다.

---

## 파일 설명

### 핵심 패키지 (`src/vix_xai/`)

| 파일                     | 대략적 라인 수 | 설명                                                                          |
| ---------------------- | -------- | --------------------------------------------------------------------------- |
| `config.py`            | ~80      | `Config` dataclass, `set_seed()`, `get_device()`                            |
| `data.py`              | ~170     | `load_frame()`, `split_by_time()`, `SequenceDataset`, `build_dataloaders()` |
| `models.py`            | ~260     | `RevIN`, `SingleTCN`, `SingleCNN`, `TCNEnsemble`, `CNNEnsemble`             |
| `training.py`          | ~100     | AMP, gradient clipping, early stopping을 포함한 `train_model()`                 |
| `eval.py`              | ~80      | `evaluate_level_rmse()`, `compute_baselines()`                              |
| `xai.py`               | ~230     | `TimeSeriesGradCAMRegression` (집계 CAM, legacy)                              |
| `event_wraping.py`     | ~450     | DTW, distributional DTW, event weighting, cost matrix                       |
| `posthoc.py`           | ~550     | matched-pair 분석, deletion test, `GradCAMEngine` (legacy)                    |
| `metrics.py`           | ~500     | reference-based DTW 지표, AUC 평가                                              |
| `concepts.py`          | ~500     | `TCAVExtractorCV`, C-DEW 분석, concept dashboard                              |
| `utils.py`             | ~150     | 시각화 보조 함수, 모델 번들 저장/불러오기                                                    |
| `experiments.py`       | ~350     | CNN 아키텍처 탐색, `run_experiment_suite()`                                       |
| **`gradcam.py`**       | ~270     | `TimeSeriesGradCAM` (target-branch), `TemporalGradientExtractor`            |
| **`tcav_temporal.py`** | ~195     | `TemporalTCAV` (CAV + dd_t + CWCR 점수화)                                      |
| **`stats.py`**         | ~336     | 8개 가설 검정, BH FDR, block bootstrap, deletion test                            |
| **`analysis.py`**      | ~428     | `run_analysis()` 통합 파이프라인 (Path A + Path B)                                 |

### 래퍼 스크립트 (`output/`)

| 파일                             | 역할                               |
| ------------------------------ | -------------------------------- |
| `vix_tcn_revin_xai_plus.py`    | 학습 + legacy XAI 실행 진입점           |
| `posthoc_analysis_v2.py`       | legacy post-hoc matched-pair 분석  |
| `cdew_concepts_v2.py`          | legacy C-DEW 개념 분석               |
| `metrics_over_time_v2.py`      | legacy reference-based DTW 지표 분석 |
| `event_warping.py`             | legacy event warping 래퍼          |
| **`gradcam_tcav_analysis.py`** | 신규 GradCAM + TCAV 분석 래퍼          |
| **`run_analysis.py`**          | 신규 전체 분석 스크립트 (notebook 스타일)     |

### 테스트 (`tests/`)

| 파일                             | 테스트 수 | 설명                                          |
| ------------------------------ | ----- | ------------------------------------------- |
| `test_smoke_train.py`          | 1     | 모델 생성, 학습, 평가, Grad-CAM                     |
| `test_event_wraping.py`        | 9     | DTW 거리 함수, event weighting                  |
| **`test_gradcam_analysis.py`** | 15    | 신규 4개 모듈 전체: stats, gradcam, tcav, pipeline |

---

## 인용

이 프레임워크를 사용하는 경우, 아래와 같이 인용할 수 있습니다.


---

## 라이선스

