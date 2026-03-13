"""
models.py — RevIN, TCN / CNN branch models, ensemble wrappers.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ═══════════════════════════════════════════════════════════════════
# RevIN
# ═══════════════════════════════════════════════════════════════════


class RevIN(nn.Module):
    """Reversible Instance Normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str, target_idx: int | None = None):
        if mode == "norm":
            self._get_stats(x)
            return self._normalize(x)
        if mode == "denorm":
            assert target_idx is not None
            return self._denormalize(x, target_idx)
        raise NotImplementedError(mode)

    def _get_stats(self, x: torch.Tensor):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, y: torch.Tensor, target_idx: int) -> torch.Tensor:
        m = self.mean[:, :, target_idx : target_idx + 1].squeeze(1)
        s = self.stdev[:, :, target_idx : target_idx + 1].squeeze(1)
        if self.affine:
            w = self.affine_weight[target_idx : target_idx + 1]
            b = self.affine_bias[target_idx : target_idx + 1]
            y = (y - b) / (w + 1e-10)
        return y * s + m


# ═══════════════════════════════════════════════════════════════════
# TCN building blocks
# ═══════════════════════════════════════════════════════════════════


class Chomp1d(nn.Module):
    """Remove trailing padding so that causal convolution output has the same length."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp].contiguous()


class TemporalBlock(nn.Module):
    """One residual block of the TCN."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        k: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            weight_norm(
                nn.Conv1d(n_in, n_out, k, stride=stride, padding=padding, dilation=dilation)
            ),
            Chomp1d(padding),
            nn.LeakyReLU(inplace=False),
            weight_norm(
                nn.Conv1d(n_out, n_out, k, stride=stride, padding=padding, dilation=dilation)
            ),
            Chomp1d(padding),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)


# ═══════════════════════════════════════════════════════════════════
# Single-branch networks
# ═══════════════════════════════════════════════════════════════════


class SingleTCN(nn.Module):
    """Single univariate TCN branch."""

    def __init__(self, channels, kernel, dropout, dilation_base):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = 1 if i == 0 else channels[i - 1]
            dil = dilation_base ** i
            pad = (kernel - 1) * dil
            layers.append(TemporalBlock(in_ch, out_ch, kernel, 1, dil, pad, dropout))
        self.network = nn.Sequential(*layers)
        self.feature_size = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, 1) → (B, C)"""
        x = x.transpose(1, 2)  # (B, 1, T)
        f = self.network(x)  # (B, C, T)
        return f[:, :, -1]  # (B, C)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Return full temporal feature map (B, C, T)."""
        return self.network(x.transpose(1, 2))


class SingleCNN(nn.Module):
    """Single univariate CNN branch."""

    def __init__(self, channels, kernel, dropout):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = 1 if i == 0 else channels[i - 1]
            pad = kernel // 2
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
                nn.LeakyReLU(inplace=False),
                nn.Dropout(dropout),
            ]
        self.network = nn.Sequential(*layers)
        self.feature_size = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        f = self.network(x)
        return f[:, :, -1]


# ═══════════════════════════════════════════════════════════════════
# Ensemble models
# ═══════════════════════════════════════════════════════════════════


class TCNEnsemble(nn.Module):
    """Per-feature TCN branches + RevIN + FC head."""

    def __init__(self, num_features: int, target_idx: int, cfg):
        super().__init__()
        self.num_features = num_features
        self.target_idx = target_idx
        self.revin = RevIN(num_features, affine=cfg.revin_affine)
        self.tcns = nn.ModuleList(
            [
                SingleTCN(cfg.tcn_channels, cfg.tcn_kernel, cfg.tcn_dropout, cfg.dilation_base)
                for _ in range(num_features)
            ]
        )
        in_dim = num_features * self.tcns[0].feature_size
        layers = []
        for h in cfg.fc_hidden:
            layers += [nn.Linear(in_dim, h), nn.LeakyReLU(inplace=False)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, "norm")
        feats = torch.cat([m(x[:, :, i : i + 1]) for i, m in enumerate(self.tcns)], dim=1)
        y = self.head(feats)
        return self.revin(y, "denorm", target_idx=self.target_idx)

    def extract_branch_embedding(
        self, x_full_normed: torch.Tensor, branch_idx: int
    ) -> torch.Tensor:
        """Extract embedding from a single TCN branch (after RevIN norm)."""
        x_single = x_full_normed[:, :, branch_idx : branch_idx + 1]
        return self.tcns[branch_idx](x_single)

    def get_all_branch_embeddings(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Return dict of branch_idx → (B, C) embeddings."""
        x_n = self.revin(x, "norm")
        return {i: self.tcns[i](x_n[:, :, i : i + 1]) for i in range(self.num_features)}


class CNNEnsemble(nn.Module):
    """Per-feature CNN branches + RevIN + FC head."""

    def __init__(self, num_features: int, target_idx: int, cfg):
        super().__init__()
        self.num_features = num_features
        self.target_idx = target_idx
        self.revin = RevIN(num_features, affine=cfg.revin_affine)
        self.cnns = nn.ModuleList(
            [
                SingleCNN(cfg.cnn_channels, cfg.cnn_kernel, cfg.cnn_dropout)
                for _ in range(num_features)
            ]
        )
        in_dim = num_features * self.cnns[0].feature_size
        layers = []
        for h in cfg.fc_hidden:
            layers += [nn.Linear(in_dim, h), nn.LeakyReLU(inplace=False)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.revin(x, "norm")
        feats = torch.cat([m(x[:, :, i : i + 1]) for i, m in enumerate(self.cnns)], dim=1)
        y = self.head(feats)
        return self.revin(y, "denorm", target_idx=self.target_idx)

    def extract_branch_embedding(
        self, x_full_normed: torch.Tensor, branch_idx: int
    ) -> torch.Tensor:
        x_single = x_full_normed[:, :, branch_idx : branch_idx + 1]
        return self.cnns[branch_idx](x_single)

    def get_all_branch_embeddings(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        x_n = self.revin(x, "norm")
        return {i: self.cnns[i](x_n[:, :, i : i + 1]) for i in range(self.num_features)}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
