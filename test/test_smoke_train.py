"""Smoke test: build model, train briefly, evaluate, run Grad-CAM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from vix_xai.config import Config, set_seed
from vix_xai.data import load_frame, build_dataloaders
from vix_xai.models import TCNEnsemble, count_parameters
from vix_xai.training import train_model
from vix_xai.eval import evaluate_level_rmse
from vix_xai.xai import TimeSeriesGradCAMRegression, collect_test_windows

def _make_csv(path, n=500):
    np.random.seed(0)
    df = pd.DataFrame({
        "날짜": pd.bdate_range("2020-01-01", periods=n),
        "SPX": 3000 + np.cumsum(np.random.randn(n)*10),
        "VIX": np.clip(15+np.cumsum(np.random.randn(n)*0.5), 8, 80),
        "Gold": 1500 + np.cumsum(np.random.randn(n)*5),
    })
    df.to_csv(path, index=False)
    return path

def test_smoke_train():
    set_seed(42)
    csv = _make_csv("/tmp/smoke.csv")
    device = torch.device("cpu")
    cfg = Config(csv_path=csv, index_col="날짜", drop_cols=(), target_col="VIX",
                 seq_len=10, batch_size=32, epochs=5, patience=3, min_epoch=2,
                 tcn_channels=(2,2), tcn_kernel=3, fc_hidden=(8,),
                 param_budget=50000, use_amp=False, num_workers=0)
    df_raw = load_frame(cfg.csv_path, cfg.index_col)
    dl_tr, dl_va, dl_te, meta = build_dataloaders(df_raw=df_raw, target_col=cfg.target_col,
        seq_len=cfg.seq_len, batch_size=cfg.batch_size, train_ratio=0.7, val_ratio=0.15,
        num_workers=0, pin_memory=False, persistent_workers=False, target_mode="level")

    model = TCNEnsemble(len(meta["feature_names"]), meta["target_index"], cfg).to(device)
    print(f"  params={count_parameters(model)}")
    model, hist = train_model(model, dl_tr, dl_va, cfg, device, nn.MSELoss(), "smoke")
    assert hist["best_val_loss"] < float("inf")

    rmse, _, _ = evaluate_level_rmse(model, dl_te, device, meta["scaler_y"],
                                      "level", meta["df_te"], cfg.target_col, cfg.seq_len)
    print(f"  RMSE={rmse:.4f}")

    X_te = collect_test_windows(dl_te)
    cam_eng = TimeSeriesGradCAMRegression(model, device=device)
    cam, _ = cam_eng.generate(X_te[0:1])
    cam_eng.remove_hooks()
    assert cam.shape[0] == cfg.seq_len
    print(f"  CAM OK shape={cam.shape}")
    print("  PASS")

if __name__ == "__main__":
    test_smoke_train()
