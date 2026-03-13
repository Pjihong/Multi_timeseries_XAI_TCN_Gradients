"""
config.py — Configuration dataclass, seed control, device selection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class Config:
    csv_path: str = "timeseries_data.csv"
    index_col: str = "날짜"
    drop_cols: Tuple[str, ...] = (
        "Silver", "Copper", "USD/GBP", "USD/CNY", "USD/JPY", "USD/EUR", "USD/CAD",
    )
    target_col: str = "VIX"

    seq_len: int = 20
    batch_size: int = 64
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    epochs: int = 300
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    patience: int = 20
    min_epoch: int = 50
    use_amp: bool = True
    deterministic: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False

    tcn_channels: Tuple[int, ...] = (3, 3)
    tcn_kernel: int = 3
    tcn_dropout: float = 0.1
    dilation_base: int = 2

    cnn_channels: Tuple[int, ...] = (8, 16)
    cnn_kernel: int = 3
    cnn_dropout: float = 0.1
    param_budget: int = 4000

    fc_hidden: Tuple[int, ...] = (16,)
    revin_affine: bool = True
    out_dir: str = "outputs"


def set_seed(seed: int = 5, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device(device: Optional[str] = None) -> torch.device:
    """Return a torch device; auto-detect CUDA if *device* is None."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
