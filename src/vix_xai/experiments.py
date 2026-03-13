"""
experiments.py — CNN architecture search and full experiment suite.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import Config, get_device, set_seed
from .data import build_dataloaders
from .eval import compute_baselines, evaluate_level_rmse
from .models import CNNEnsemble, TCNEnsemble, count_parameters
from .training import train_model
from .utils import (
    _build_model_from_snapshot,
    plot_losses,
    plot_predictions,
    plot_revin_params,
    save_model_bundle,
)


# ═══════════════════════════════════════════════════════════════════
# CNN budget search (val-based)
# ═══════════════════════════════════════════════════════════════════


def search_cnn_config_under_budget(
    cfg: Config,
    num_features: int,
    target_idx: int,
    device: torch.device,
    dl_tr=None,
    dl_va=None,
    criterion=None,
    quick_epochs: int = 30,
) -> Tuple[Optional[dict], List[dict]]:
    """
    Search CNN config under param budget.

    If *dl_tr* / *dl_va* / *criterion* are provided, train briefly and select by
    val loss.  Otherwise fall back to max-params heuristic.
    """
    channel_candidates = [(8, 8), (8, 16), (16, 16), (16, 32), (32, 32)]
    kernel_candidates = [3, 5]
    search_log: List[dict] = []

    best_cfg_result: Optional[dict] = None
    best_val = float("inf")
    best_n_params_fallback = -1

    use_val = dl_tr is not None and dl_va is not None and criterion is not None

    for chs in channel_candidates:
        for k in kernel_candidates:
            tmp = copy.deepcopy(cfg)
            tmp.cnn_channels = chs
            tmp.cnn_kernel = k
            m = CNNEnsemble(num_features, target_idx, tmp).to(device)
            n_params, _ = count_parameters(m)

            over_budget = n_params > cfg.param_budget
            val_loss = float("nan")

            if over_budget:
                search_log.append(
                    {
                        "cnn_channels": str(chs),
                        "cnn_kernel": k,
                        "n_params": n_params,
                        "val_loss_best": float("nan"),
                        "val_rmse_level": float("nan"),
                        "over_budget": True,
                        "selected": False,
                    }
                )
                del m
                continue

            if use_val:
                tmp_cfg = copy.deepcopy(cfg)
                tmp_cfg.epochs = quick_epochs
                tmp_cfg.min_epoch = 5
                tmp_cfg.patience = 10
                m, hist = train_model(
                    m, dl_tr, dl_va, tmp_cfg, device, criterion,
                    model_name=f"cnn_search_{chs}_{k}",
                )
                val_loss = hist["best_val_loss"]
                if val_loss < best_val:
                    best_val = val_loss
                    best_cfg_result = {
                        "cnn_channels": chs,
                        "cnn_kernel": k,
                        "n_params": n_params,
                        "val_loss_best": val_loss,
                    }
            else:
                if n_params > best_n_params_fallback:
                    best_n_params_fallback = n_params
                    best_cfg_result = {
                        "cnn_channels": chs,
                        "cnn_kernel": k,
                        "n_params": n_params,
                        "val_loss_best": float("nan"),
                    }

            search_log.append(
                {
                    "cnn_channels": str(chs),
                    "cnn_kernel": k,
                    "n_params": n_params,
                    "val_loss_best": val_loss,
                    "val_rmse_level": float("nan"),
                    "over_budget": False,
                    "selected": False,
                }
            )
            del m
            torch.cuda.empty_cache()

    if best_cfg_result is not None:
        for row in search_log:
            if (
                row["cnn_channels"] == str(best_cfg_result["cnn_channels"])
                and row["cnn_kernel"] == best_cfg_result["cnn_kernel"]
            ):
                row["selected"] = True
        print(
            f"[CNN budget] channels={best_cfg_result['cnn_channels']} "
            f"kernel={best_cfg_result['cnn_kernel']} "
            f"params={best_cfg_result['n_params']} "
            f"val={best_cfg_result.get('val_loss_best', 'N/A')}"
        )
    else:
        print(
            f"[CNN budget] WARNING: ALL candidates exceed param_budget={cfg.param_budget}. "
            f"Using default cfg."
        )

    return best_cfg_result, search_log


# ═══════════════════════════════════════════════════════════════════
# Main experiment suite
# ═══════════════════════════════════════════════════════════════════


def run_experiment_suite(
    cfg: Config,
    df_raw: pd.DataFrame,
    experiment_settings=None,
    architectures=None,
    seeds: Tuple[int, ...] = (5,),
    device=None,
) -> dict:
    """
    Run the full experiment suite.

    Model / TCN selection is based on **validation RMSE** (not test).
    """
    exp_dir = os.path.join(cfg.out_dir, "experiments")
    bun_dir = os.path.join(cfg.out_dir, "bundles")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(bun_dir, exist_ok=True)

    device = device or get_device()
    experiment_settings = experiment_settings or [
        ("level", "mse", "none"),
        ("level", "huber", "none"),
        ("diff", "mse", "none"),
        ("diff", "huber", "none"),
        ("log", "mse", "none"),
        ("log", "huber", "none"),
        ("level", "mse", "softplus"),
    ]
    architectures = architectures or ["tcn", "cnn"]

    # ── save config snapshot ──
    config_snap = {
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "train_ratio": cfg.train_ratio,
        "val_ratio": cfg.val_ratio,
        "target_col": cfg.target_col,
        "drop_cols": list(cfg.drop_cols),
        "architectures": architectures,
        "experiment_settings": [list(s) for s in experiment_settings],
        "selection_metric": "val_rmse_level",
        "seeds": list(seeds),
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "param_budget": cfg.param_budget,
    }
    with open(os.path.join(exp_dir, "config_snapshot.json"), "w") as f:
        json.dump(config_snap, f, indent=2, ensure_ascii=False)

    per_seed_rows: list = []
    cnn_search_rows: list = []
    capacity_rows: list = []
    baseline_rows: list = []
    mode_alignment_rows: list = []

    all_results: dict = {}
    _cnn_search_cache: dict = {}
    _splits_saved = False
    _mode_alignment_seen: set = set()

    for seed in seeds:
        set_seed(seed, deterministic=cfg.deterministic)

        for target_mode, loss_type, out_act in experiment_settings:
            dl_tr, dl_va, dl_te, meta = build_dataloaders(
                df_raw=df_raw,
                target_col=cfg.target_col,
                seq_len=cfg.seq_len,
                batch_size=cfg.batch_size,
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                num_workers=cfg.num_workers,
                pin_memory=(cfg.pin_memory and device.type == "cuda"),
                persistent_workers=cfg.persistent_workers,
                target_mode=target_mode,
            )
            num_features = len(meta["feature_names"])
            target_idx = meta["target_index"]

            # mode alignment report
            if seed == seeds[0] and target_mode not in _mode_alignment_seen:
                _mode_alignment_seen.add(target_mode)
                mode_alignment_rows.append(
                    {
                        "target_mode": target_mode,
                        "model_target_col": meta["model_target_col"],
                        "first_valid_date": str(meta["df_tr"].index[0]),
                        "last_valid_date": str(meta["df_te"].index[-1]),
                        "n_samples_total": len(meta["df_tr"])
                        + len(meta["df_va"])
                        + len(meta["df_te"]),
                        "n_train": len(meta["df_tr"]),
                        "n_val": len(meta["df_va"]),
                        "n_test": len(meta["df_te"]),
                    }
                )

            # splits (save once)
            if not _splits_saved:
                _splits_saved = True
                splits = {
                    "train_start": str(meta["df_tr"].index[0]),
                    "train_end": str(meta["df_tr"].index[-1]),
                    "val_start": str(meta["df_va"].index[0]),
                    "val_end": str(meta["df_va"].index[-1]),
                    "test_start": str(meta["df_te"].index[0]),
                    "test_end": str(meta["df_te"].index[-1]),
                    "n_total": len(meta["df_tr"])
                    + len(meta["df_va"])
                    + len(meta["df_te"]),
                    "n_train": len(meta["df_tr"]),
                    "n_val": len(meta["df_va"]),
                    "n_test": len(meta["df_te"]),
                }
                with open(os.path.join(exp_dir, "splits.json"), "w") as f:
                    json.dump(splits, f, indent=2, ensure_ascii=False)

            # baselines (once per seed)
            if (seed, "baseline") not in _cnn_search_cache:
                _cnn_search_cache[(seed, "baseline")] = True
                bl = compute_baselines(meta, cfg, dl_va, dl_te)
                for b in bl:
                    b["seed"] = seed
                    b["target_mode"] = target_mode
                    baseline_rows.append(b)

            # CNN search
            cnn_cache_key = (seed, num_features)
            if cnn_cache_key not in _cnn_search_cache:
                criterion_search = (
                    nn.MSELoss() if loss_type == "mse" else nn.SmoothL1Loss(beta=1.0)
                )
                cnn_best, cnn_log = search_cnn_config_under_budget(
                    cfg,
                    num_features,
                    target_idx,
                    device,
                    dl_tr=dl_tr,
                    dl_va=dl_va,
                    criterion=criterion_search,
                    quick_epochs=30,
                )
                _cnn_search_cache[cnn_cache_key] = (cnn_best, cnn_log)
                for row in cnn_log:
                    row.update(
                        {
                            "seed": seed,
                            "target_mode": target_mode,
                            "loss_type": loss_type,
                            "out_act": out_act,
                        }
                    )
                    cnn_search_rows.append(row)
            else:
                cnn_best, _ = _cnn_search_cache[cnn_cache_key]

            if cnn_best is not None:
                cfg.cnn_channels = cnn_best["cnn_channels"]
                cfg.cnn_kernel = cnn_best["cnn_kernel"]

            for arch in architectures:
                if arch == "tcn":
                    model = TCNEnsemble(num_features, target_idx, cfg).to(device)
                else:
                    model = CNNEnsemble(num_features, target_idx, cfg).to(device)
                if out_act == "softplus":
                    model.head.add_module("softplus", nn.Softplus())

                n_total, n_train = count_parameters(model)
                criterion = (
                    nn.MSELoss() if loss_type == "mse" else nn.SmoothL1Loss(beta=1.0)
                )
                mname = f"{arch}_{target_mode}_{loss_type}_{out_act}_s{seed}"

                model, hist = train_model(model, dl_tr, dl_va, cfg, device, criterion, mname)

                # val RMSE (selection metric)
                val_rmse, _, _ = evaluate_level_rmse(
                    model,
                    dl_va,
                    device,
                    meta["scaler_y"],
                    meta["target_mode"],
                    meta["df_va"],
                    meta["target_col_original"],
                    cfg.seq_len,
                )

                # test RMSE (reporting only)
                test_rmse, preds_lv, trues_lv = evaluate_level_rmse(
                    model,
                    dl_te,
                    device,
                    meta["scaler_y"],
                    meta["target_mode"],
                    meta["df_te"],
                    meta["target_col_original"],
                    cfg.seq_len,
                )

                key = (seed, target_mode, loss_type, out_act, arch)
                snap = {
                    "arch": arch,
                    "cfg": asdict(cfg),
                    "num_features": num_features,
                    "target_idx": target_idx,
                    "out_act": out_act,
                    "target_mode": target_mode,
                    "loss_type": loss_type,
                }

                all_results[key] = {
                    "val_rmse": val_rmse,
                    "test_rmse": test_rmse,
                    "best_val_loss": hist["best_val_loss"],
                    "history": hist,
                    "snapshot": snap,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "meta": meta,
                    "preds_lv": preds_lv,
                    "trues_lv": trues_lv,
                }

                per_seed_rows.append(
                    {
                        "seed": seed,
                        "arch": arch,
                        "target_mode": target_mode,
                        "loss_type": loss_type,
                        "out_act": out_act,
                        "best_val_loss": hist["best_val_loss"],
                        "val_rmse_level": val_rmse,
                        "test_rmse_level": test_rmse,
                        "selected_best_model": False,
                        "selected_best_tcn": False,
                    }
                )
                capacity_rows.append(
                    {
                        "arch": arch,
                        "target_mode": target_mode,
                        "loss_type": loss_type,
                        "out_act": out_act,
                        "n_params": n_total,
                        "trainable_params": n_train,
                        "approx_flops": "N/A",
                        "train_time_sec": hist["total_time_sec"],
                    }
                )

                print(f"  [{mname}] val_rmse={val_rmse:.4f} test_rmse={test_rmse:.4f}")
                del model
                torch.cuda.empty_cache()

    # ── SELECT BEST BY VAL ──
    best_key, best_val_rmse = None, float("inf")
    best_tcn_key, best_tcn_val = None, float("inf")

    for k, v in all_results.items():
        if v["val_rmse"] < best_val_rmse:
            best_val_rmse = v["val_rmse"]
            best_key = k
        if k[4] == "tcn" and v["val_rmse"] < best_tcn_val:
            best_tcn_val = v["val_rmse"]
            best_tcn_key = k

    for row in per_seed_rows:
        rk = (row["seed"], row["target_mode"], row["loss_type"], row["out_act"], row["arch"])
        if rk == best_key:
            row["selected_best_model"] = True
        if rk == best_tcn_key:
            row["selected_best_tcn"] = True

    # ── SAVE CSVs ──
    pd.DataFrame(per_seed_rows).to_csv(os.path.join(exp_dir, "per_seed_results.csv"), index=False)
    pd.DataFrame(cnn_search_rows).to_csv(os.path.join(exp_dir, "cnn_search_log.csv"), index=False)
    pd.DataFrame(capacity_rows).to_csv(os.path.join(exp_dir, "model_capacity.csv"), index=False)
    pd.DataFrame(baseline_rows).to_csv(os.path.join(exp_dir, "baseline_metrics.csv"), index=False)
    pd.DataFrame(mode_alignment_rows).to_csv(
        os.path.join(exp_dir, "mode_alignment_report.csv"), index=False
    )

    # ── SUMMARY ──
    df_per = pd.DataFrame(per_seed_rows)
    summary_rows = []
    for (arch, tm, lt, oa), grp in df_per.groupby(
        ["arch", "target_mode", "loss_type", "out_act"]
    ):
        summary_rows.append(
            {
                "arch": arch,
                "target_mode": tm,
                "loss_type": lt,
                "out_act": oa,
                "val_rmse_mean": grp["val_rmse_level"].mean(),
                "val_rmse_std": grp["val_rmse_level"].std(),
                "test_rmse_mean": grp["test_rmse_level"].mean(),
                "test_rmse_std": grp["test_rmse_level"].std(),
                "n_seeds": len(grp),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("val_rmse_mean").reset_index(drop=True)
    summary_df.to_csv(os.path.join(exp_dir, "summary_val.csv"), index=False)

    # ── RECONSTRUCT BEST MODELS ──
    best_r = all_results[best_key]
    best_model = _build_model_from_snapshot(best_r["snapshot"], best_r["state_dict"], device)
    best_meta = best_r["meta"]
    best_preds = best_r["preds_lv"]
    best_trues = best_r["trues_lv"]

    best_tcn_r = all_results[best_tcn_key] if best_tcn_key else None
    best_tcn_model = (
        _build_model_from_snapshot(best_tcn_r["snapshot"], best_tcn_r["state_dict"], device)
        if best_tcn_r
        else None
    )

    # ── SAVE BUNDLES ──
    save_model_bundle(
        os.path.join(bun_dir, "best_model_bundle.pt"),
        best_r["snapshot"],
        best_r["state_dict"],
        best_meta,
    )
    if best_tcn_r:
        save_model_bundle(
            os.path.join(bun_dir, "best_tcn_bundle.pt"),
            best_tcn_r["snapshot"],
            best_tcn_r["state_dict"],
            best_tcn_r["meta"],
        )

    # ── FINAL TEST METRICS ──
    bl_df = pd.DataFrame(baseline_rows)
    bl_test = bl_df[bl_df["split"] == "test"]
    final_metrics = {
        "best_key_by_val": list(best_key),
        "best_tcn_key_by_val": list(best_tcn_key) if best_tcn_key else None,
        "final_test_rmse_best_model": best_r["test_rmse"],
        "final_test_rmse_best_tcn": best_tcn_r["test_rmse"] if best_tcn_r else None,
        "baseline_comparison": bl_test.to_dict(orient="records") if len(bl_test) > 0 else [],
    }
    with open(os.path.join(exp_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    # ── PREDICTIONS CSV ──
    dates_te = best_meta["df_te"].index[cfg.seq_len:]
    n = min(len(dates_te), len(best_preds))
    pred_df = pd.DataFrame(
        {
            "date": dates_te[:n],
            "y_true": best_trues[:n],
            "y_pred": best_preds[:n],
            "split": "test",
            "model_key": str(best_key),
        }
    )
    pred_df.to_csv(os.path.join(exp_dir, "predictions_best_test.csv"), index=False)

    # ── PLOTS ──
    plot_losses(
        best_r["history"],
        title=f"Loss: {best_key}",
        savepath=os.path.join(exp_dir, "loss_curve_best_model.png"),
    )
    if best_tcn_r:
        plot_losses(
            best_tcn_r["history"],
            title=f"Loss: {best_tcn_key}",
            savepath=os.path.join(exp_dir, "loss_curve_best_tcn.png"),
        )
    plot_predictions(
        best_trues[:n],
        best_preds[:n],
        title=f"Pred vs True: {best_key}",
        savepath=os.path.join(exp_dir, "pred_vs_true_best_model.png"),
    )
    plot_revin_params(
        best_model,
        best_meta["feature_names"],
        save_path=os.path.join(exp_dir, "revin_params_best_model.png"),
    )

    return {
        "all_results": all_results,
        "summary_df": summary_df,
        "best_key": best_key,
        "best_val_rmse": best_val_rmse,
        "best_test_rmse": best_r["test_rmse"],
        "best_model": best_model,
        "best_meta": best_meta,
        "best_preds_lv": best_preds,
        "best_trues_lv": best_trues,
        "best_tcn_key": best_tcn_key,
        "best_tcn_val_rmse": best_tcn_val,
        "best_tcn_test_rmse": best_tcn_r["test_rmse"] if best_tcn_r else None,
        "best_tcn_model": best_tcn_model,
        "best_tcn_meta": best_tcn_r["meta"] if best_tcn_r else None,
        "device": device,
    }
