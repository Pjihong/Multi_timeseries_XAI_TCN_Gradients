"""
vix_xai — VIX TCN + XAI + Event-Warping + C-DEW (Concept-DTW) package.
"""

from .config import Config, set_seed, get_device
from .data import load_frame, split_by_time, transform_for_model, SequenceDataset, build_dataloaders
from .models import (
    RevIN, Chomp1d, TemporalBlock, SingleTCN, TCNEnsemble,
    SingleCNN, CNNEnsemble, count_parameters,
)
from .training import EarlyStopping, train_model
from .eval import evaluate_level_rmse, compute_baselines
from .xai import (
    TimeSeriesGradCAMRegression, collect_test_windows,
    inverse_all_X_windows, extract_multivariate_embeddings,
    evaluate_cpd_performance,
)
from .utils import (
    plot_losses, plot_predictions, plot_revin_params,
    save_model_bundle, load_model_bundle,
)
from .experiments import search_cnn_config_under_budget, run_experiment_suite

__all__ = [
    # config
    "Config", "set_seed", "get_device",
    # data
    "load_frame", "split_by_time", "transform_for_model",
    "SequenceDataset", "build_dataloaders",
    # models
    "RevIN", "Chomp1d", "TemporalBlock", "SingleTCN", "TCNEnsemble",
    "SingleCNN", "CNNEnsemble", "count_parameters",
    # training
    "EarlyStopping", "train_model",
    # eval
    "evaluate_level_rmse", "compute_baselines",
    # xai
    "TimeSeriesGradCAMRegression", "collect_test_windows",
    "inverse_all_X_windows", "extract_multivariate_embeddings",
    "evaluate_cpd_performance",
    # utils
    "plot_losses", "plot_predictions", "plot_revin_params",
    "save_model_bundle", "load_model_bundle",
    # experiments
    "search_cnn_config_under_budget", "run_experiment_suite",
]
