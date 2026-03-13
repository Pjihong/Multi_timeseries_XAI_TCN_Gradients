"""
Wrapper for GradCAM + Temporal TCAV analysis.

Usage:
    python gradcam_tcav_analysis.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vix_xai.analysis import run_analysis  # noqa: F401
from vix_xai.gradcam import (  # noqa: F401
    TimeSeriesGradCAM,
    TemporalGradientExtractor,
    extract_embeddings,
    resolve_target_branch,
)
from vix_xai.tcav_temporal import TemporalTCAV  # noqa: F401
from vix_xai.stats import (  # noqa: F401
    two_sample_perm,
    paired_bootstrap_auc,
    alignment_perm,
    paired_perm,
    deletion_test,
    accuracy_above_chance,
    interaction_perm,
    cosine_stability,
    block_bootstrap_ci,
    benjamini_hochberg,
    subsample_nonoverlap,
)
