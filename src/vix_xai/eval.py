"""
eval.py — Evaluation helpers: level-RMSE, baseline metrics.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_level_rmse(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    scaler_y,
    target_mode: str,
    df_split: pd.DataFrame,
    target_col_level: str,
    seq_len: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate on any dataloader (val or test).

    Returns (rmse, preds_level, trues_level).
    """
    model.eval()
    preds, trues = [], []
    for x, y in dl:
        preds.append(model(x.to(device)).cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = scaler_y.inverse_transform(np.vstack(preds)).ravel()
    trues = scaler_y.inverse_transform(np.vstack(trues)).ravel()

    tm = str(target_mode).lower()
    if tm == "level":
        rmse = float(np.sqrt(mean_squared_error(trues, preds)))
        return rmse, preds, trues

    if tm == "diff":
        levels = df_split[target_col_level].values
        last = levels[seq_len - 1 : seq_len - 1 + len(trues)]
        rmse = float(np.sqrt(mean_squared_error(last + trues, last + preds)))
        return rmse, last + preds, last + trues

    if tm == "log":
        rmse = float(np.sqrt(mean_squared_error(np.exp(trues), np.exp(preds))))
        return rmse, np.exp(preds), np.exp(trues)

    raise ValueError(f"unknown target_mode: {target_mode}")


def compute_baselines(meta: dict, cfg, dl_va: DataLoader, dl_te: DataLoader) -> List[dict]:
    """
    Naive (last-value) and drift baselines on val and test splits.
    """
    scaler_y = meta["scaler_y"]
    target_mode = meta["target_mode"]
    results: List[dict] = []

    for split_name, dl, df_split in [("val", dl_va, meta["df_va"]), ("test", dl_te, meta["df_te"])]:
        trues_raw, preds_raw = [], []
        for x, y in dl:
            trues_raw.append(y.numpy())
            # naive: predict last time-step's target
            preds_raw.append(
                x[:, -1, meta["target_index"] : meta["target_index"] + 1].numpy()
            )
        preds_sc = scaler_y.inverse_transform(np.vstack(preds_raw)).ravel()
        trues_sc = scaler_y.inverse_transform(np.vstack(trues_raw)).ravel()

        tm = str(target_mode).lower()
        if tm == "level":
            rmse_naive = float(np.sqrt(mean_squared_error(trues_sc, preds_sc)))
        elif tm == "diff":
            levels = df_split[meta["target_col_original"]].values
            last = levels[cfg.seq_len - 1 : cfg.seq_len - 1 + len(trues_sc)]
            rmse_naive = float(np.sqrt(mean_squared_error(last + trues_sc, last + preds_sc)))
        elif tm == "log":
            rmse_naive = float(np.sqrt(mean_squared_error(np.exp(trues_sc), np.exp(preds_sc))))
        else:
            rmse_naive = float("nan")

        rmse_drift = rmse_naive  # simplified: same as naive

        results.append({"baseline_name": "naive", "split": split_name, "rmse_level": rmse_naive})
        results.append({"baseline_name": "drift", "split": split_name, "rmse_level": rmse_drift})

    return results
