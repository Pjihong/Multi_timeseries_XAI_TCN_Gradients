"""
utils.py — Plotting, model bundle save / load.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .config import Config, get_device
from .models import CNNEnsemble, TCNEnsemble


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════


def plot_losses(
    history: dict,
    title: str = "loss",
    savepath: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    fig = plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_predictions(
    trues: np.ndarray,
    preds: np.ndarray,
    title: str = "Prediction vs Truth",
    ylabel: str = "VIX",
    savepath: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    fig = plt.figure(figsize=(12, 4))
    plt.plot(trues, label="true", linewidth=2)
    plt.plot(preds, label="pred", alpha=0.8)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        plt.savefig(savepath, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_revin_params(
    model: nn.Module,
    feature_names: list,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    if not hasattr(model, "revin") or not model.revin.affine:
        return
    w = model.revin.affine_weight.detach().cpu().numpy()
    b = model.revin.affine_bias.detach().cpu().numpy()
    x = np.arange(len(feature_names))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.bar(x, w)
    ax1.axhline(1.0, ls="--")
    ax1.set_title("RevIN affine weights")
    ax1.grid(axis="y", alpha=0.3)
    ax2.bar(x, b)
    ax2.axhline(0.0)
    ax2.set_title("RevIN affine bias")
    ax2.set_xticks(x)
    ax2.set_xticklabels(feature_names, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Model bundle save / load
# ═══════════════════════════════════════════════════════════════════


def _build_model_from_snapshot(
    snapshot: dict,
    state_dict: dict,
    device: torch.device,
) -> nn.Module:
    """Reconstruct a model from its snapshot metadata and state dict."""
    cfg_obj = snapshot["cfg"]
    cfg = Config(**cfg_obj) if isinstance(cfg_obj, dict) else cfg_obj
    num_features = snapshot["num_features"]
    target_idx = snapshot["target_idx"]
    out_act = snapshot["out_act"]
    arch = snapshot["arch"]

    if arch == "tcn":
        m = TCNEnsemble(num_features, target_idx, cfg)
    elif arch == "cnn":
        m = CNNEnsemble(num_features, target_idx, cfg)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    if out_act == "softplus":
        m.head.add_module("softplus", nn.Softplus())
    m.load_state_dict(state_dict)
    return m.to(device)


def save_model_bundle(
    path: str,
    snapshot: dict,
    state_dict: dict,
    meta: dict,
) -> None:
    """Persist model snapshot + weights + metadata to a .pt file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta_save = {k: v for k, v in meta.items() if k not in ("df_tr", "df_va", "df_te")}
    torch.save({"snapshot": snapshot, "state_dict": state_dict, "meta": meta_save}, path)


def load_model_bundle(
    path: str,
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, dict, dict]:
    device = device or get_device()
    bundle = torch.load(path, map_location=device, weights_only=False)
    model = _build_model_from_snapshot(bundle["snapshot"], bundle["state_dict"], device)
    model.eval()
    return model, bundle.get("meta", {}), bundle["snapshot"]