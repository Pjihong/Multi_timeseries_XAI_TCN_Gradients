"""
training.py — Training loop, early stopping, validation loss.
"""

from __future__ import annotations

import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class EarlyStopping:
    """Patience-based early stopping (waits until *min_epoch*)."""

    def __init__(self, patience: int = 20, min_epoch: int = 50, delta: float = 0.0):
        self.patience = patience
        self.min_epoch = min_epoch
        self.delta = delta
        self.best: float | None = None
        self.count: int = 0
        self.stop: bool = False

    def step(self, val_loss: float, epoch: int) -> None:
        if epoch < self.min_epoch:
            return
        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True


@torch.no_grad()
def _val_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute average validation loss."""
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        b = x.size(0)
        total += float(loss.item()) * b
        n += b
    return total / max(n, 1)


def train_model(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    cfg,
    device: torch.device,
    criterion: nn.Module,
    model_name: str,
) -> tuple[nn.Module, dict]:
    """
    Full training loop with AMP, gradient clipping, LR scheduler, and early stopping.

    Returns the trained model (with best val weights) and a history dict.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=cfg.patience, min_lr=1e-6,
    )
    es = EarlyStopping(patience=cfg.patience, min_epoch=cfg.min_epoch)

    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    hist: dict = {"train_loss": [], "val_loss": [], "lr": []}
    best_state, best_val = None, float("inf")
    t0 = time.time()
    it = tqdm(range(1, cfg.epochs + 1), desc=f"Train {model_name}", unit="epoch")

    for epoch in it:
        model.train()
        total, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if cfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            b = x.size(0)
            total += float(loss.item()) * b
            n += b

        tr_loss = total / max(n, 1)
        va_loss = _val_loss(model, dl_va, criterion, device)
        sch.step(va_loss)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["lr"].append(opt.param_groups[0]["lr"])
        it.set_postfix(train=f"{tr_loss:.4f}", val=f"{va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
        es.step(va_loss, epoch)
        if es.stop:
            it.write(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["total_time_sec"] = time.time() - t0
    hist["best_val_loss"] = float(best_val)
    print(f"[{model_name}] time={hist['total_time_sec']:.1f}s best_val={best_val:.4f}")
    return model, hist
