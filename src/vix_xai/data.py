"""
data.py — Data loading, splitting, transformation, Dataset, DataLoader builders.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════
# Loading & splitting
# ═══════════════════════════════════════════════════════════════════


def load_frame(
    csv_path: str,
    index_col: str,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load a CSV, parse dates, drop columns, remove NaN rows."""
    df = pd.read_csv(csv_path)
    if index_col in df.columns:
        df[index_col] = pd.to_datetime(df[index_col])
        df = df.set_index(index_col).sort_index()
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df.dropna()


def split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / val / test split."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = n_train + int(n * val_ratio)
    return df.iloc[:n_train], df.iloc[n_train:n_val], df.iloc[n_val:]


# ═══════════════════════════════════════════════════════════════════
# Feature engineering
# ═══════════════════════════════════════════════════════════════════


def transform_for_model(
    df: pd.DataFrame,
    target_col: str,
    target_lags: int,
    level_keep_base: List[str],
    target_mode: str = "level",
) -> Tuple[pd.DataFrame, str]:
    """
    Transform raw dataframe according to *target_mode*.

    Returns
    -------
    out : pd.DataFrame
        Transformed features + model target column.
    model_target_col : str
        Name of the target column inside *out*.
    """
    level_keep = [c for c in level_keep_base if c in df.columns and c != target_col]
    eps = 1e-12

    if target_mode == "level":
        model_target_col = target_col
        cols_exclude = set(level_keep + [model_target_col])
        logret_cols = [c for c in df.columns if c not in cols_exclude]
        df_logret = np.log(df[logret_cols].clip(lower=eps)).diff()
        out = pd.concat([df[level_keep], df[[model_target_col]], df_logret], axis=1)

    elif target_mode == "diff":
        model_target_col = f"{target_col}_diff"
        df[model_target_col] = df[target_col].diff()
        cols_exclude = set(level_keep + [target_col, model_target_col])
        logret_cols = [c for c in df.columns if c not in cols_exclude]
        df_logret = np.log(df[logret_cols].clip(lower=eps)).diff()
        out = pd.concat(
            [df[level_keep], df[[target_col]], df[[model_target_col]], df_logret],
            axis=1,
        )

    elif target_mode == "log":
        model_target_col = f"log_{target_col}"
        df[model_target_col] = np.log(df[target_col].clip(lower=eps))
        cols_exclude = set(level_keep + [model_target_col])
        logret_cols = [c for c in df.columns if c not in cols_exclude]
        df_logret = np.log(df[logret_cols].clip(lower=eps)).diff()
        out = pd.concat([df[level_keep], df[[model_target_col]], df_logret], axis=1)

    else:
        raise ValueError(f"unknown target_mode: {target_mode}")

    if target_lags:
        for k in range(1, target_lags + 1):
            out[f"{model_target_col}_lag{k}"] = df[model_target_col].shift(k)

    return out.dropna(), model_target_col


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════


class SequenceDataset(Dataset):
    """Sliding-window dataset for time series."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        xs, ys = [], []
        for i in range(len(X) - seq_len):
            xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        self.X = np.asarray(xs, dtype=np.float32)
        self.y = np.asarray(ys, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════
# DataLoader builder
# ═══════════════════════════════════════════════════════════════════


def build_dataloaders(
    df_raw: pd.DataFrame,
    target_col: str,
    seq_len: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    target_lags: int = 0,
    level_keep_base: Optional[List[str]] = None,
    target_mode: str = "level",
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train / val / test DataLoaders and return metadata dict.
    """
    level_keep_base = level_keep_base or []
    df_model, model_target_col = transform_for_model(
        df=df_raw.copy(),
        target_col=target_col,
        target_lags=target_lags,
        level_keep_base=level_keep_base,
        target_mode=target_mode,
    )
    assert len(df_model) > seq_len + 5

    df_tr, df_va, df_te = split_by_time(df_model, train_ratio, val_ratio)
    feature_names = list(df_model.columns)
    X_tr, X_va, X_te = df_tr.values, df_va.values, df_te.values
    y_idx = feature_names.index(model_target_col)

    y_tr = X_tr[:, [y_idx]]
    y_va = X_va[:, [y_idx]]
    y_te = X_te[:, [y_idx]]

    scaler_X = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr)

    X_tr = scaler_X.transform(X_tr)
    X_va = scaler_X.transform(X_va)
    X_te = scaler_X.transform(X_te)
    y_tr = scaler_y.transform(y_tr)
    y_va = scaler_y.transform(y_va)
    y_te = scaler_y.transform(y_te)

    ds_tr = SequenceDataset(X_tr, y_tr, seq_len)
    ds_va = SequenceDataset(X_va, y_va, seq_len)
    ds_te = SequenceDataset(X_te, y_te, seq_len)

    pw = persistent_workers and num_workers > 0
    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=False,
        pin_memory=pin_memory, persistent_workers=pw,
    )
    dl_va = DataLoader(
        ds_va, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
        pin_memory=pin_memory, persistent_workers=pw,
    )
    dl_te = DataLoader(
        ds_te, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
        pin_memory=pin_memory, persistent_workers=pw,
    )

    meta = dict(
        feature_names=feature_names,
        target_col_original=target_col,
        model_target_col=model_target_col,
        target_mode=target_mode,
        target_index=y_idx,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        df_tr=df_tr,
        df_va=df_va,
        df_te=df_te,
    )
    return dl_tr, dl_va, dl_te, meta
