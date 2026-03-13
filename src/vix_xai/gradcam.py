
"""
gradcam.py — Target-branch TimeSeriesGradCAM + TemporalGradientExtractor.

v3 변경:
  - register_full_backward_hook 제거 → tensor register_hook 사용
  - forward hook에서 clone() 반환 → LeakyReLU(inplace=True) 충돌 방지
  - resolve_target_branch: meta["target_index"] 우선
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def get_branches(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    for attr in ("tcns", "cnns"):
        b = getattr(model, attr, None)
        if b is not None and len(b) > 0:
            return attr, b
    raise RuntimeError("Model has neither 'tcns' nor 'cnns'")


def get_last_conv(branch: nn.Module) -> Optional[nn.Conv1d]:
    last = None
    for m in branch.modules():
        if isinstance(m, nn.Conv1d):
            last = m
    return last


def resolve_target_branch(model: nn.Module, meta: dict, cfg) -> int:
    """
    모델의 target branch index.
    meta["target_index"]를 우선 사용.
    target_mode="log"이면 features=['log_VIX',...,'VIX',...]가 되어
    cfg.target_col="VIX"가 잘못된 index를 줄 수 있으므로.
    """
    if "target_index" in meta:
        return int(meta["target_index"])
    names = meta.get("feature_names", [])
    mtc = meta.get("model_target_col", "")
    if mtc in names:
        return names.index(mtc)
    if hasattr(cfg, "target_col") and cfg.target_col in names:
        return names.index(cfg.target_col)
    return 0


class TimeSeriesGradCAM:
    """
    Branch-specific temporal Grad-CAM.

    hook 방식 (v3):
      forward hook에서 output.clone() 반환 → view chain 끊기
      → clone된 tensor에 tensor-level register_hook → gradient 캡처
      → register_full_backward_hook 사용하지 않음 (in-place 충돌 방지)
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        _, self.branches = get_branches(model)
        self._fmap: Optional[np.ndarray] = None
        self._grad: Optional[np.ndarray] = None
        self._handles: List = []
        self._hooked_idx: Optional[int] = None

    def _ensure_hooked(self, branch_idx: int):
        if self._hooked_idx == branch_idx:
            return
        self._remove()
        conv = get_last_conv(self.branches[branch_idx])
        if conv is None:
            raise RuntimeError(f"No Conv1d in branch {branch_idx}")
        parent = self

        def fwd_hook(m, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            parent._fmap = o.detach().clone().cpu().numpy()
            cloned = o.clone()
            cloned.register_hook(
                lambda g: setattr(parent, '_grad', g.detach().clone().cpu().numpy())
            )
            return cloned

        self._handles.append(conv.register_forward_hook(fwd_hook))
        self._hooked_idx = branch_idx

    def _remove(self):
        for h in self._handles:
            try: h.remove()
            except: pass
        self._handles.clear()
        self._hooked_idx = None

    def cleanup(self):
        self._remove()

    def __del__(self):
        self._remove()

    def _compute_cam(self, T: int) -> np.ndarray:
        if self._fmap is None or self._grad is None:
            return np.zeros(T)
        A = self._fmap[0]  # (C, T')
        G = self._grad[0]  # (C, T')
        w = G.mean(axis=-1)
        cam = (w[:, None] * A).sum(axis=0)
        if cam.size != T:
            cam = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, cam.size), cam)
        return cam

    def generate(
        self, x: torch.Tensor, branch_idx: int,
        smooth_steps: int = 0, noise_sigma: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert x.dim() == 3 and x.size(0) == 1
        x = x.to(self.device)
        T = x.size(1)
        self._ensure_hooked(branch_idx)
        was = self.model.training
        self.model.eval()
        try:
            if smooth_steps > 0:
                cams = []
                for _ in range(smooth_steps):
                    self._fmap = self._grad = None
                    xi = (x + torch.randn_like(x) * noise_sigma).requires_grad_(True)
                    self.model.zero_grad(set_to_none=True)
                    self.model(xi).squeeze().backward()
                    cams.append(self._compute_cam(T))
                raw = np.mean(cams, axis=0)
            else:
                self._fmap = self._grad = None
                xi = x.clone().requires_grad_(True)
                self.model.zero_grad(set_to_none=True)
                self.model(xi).squeeze().backward()
                raw = self._compute_cam(T)
            mx = np.abs(raw).max()
            s = raw / mx if mx > 0 else raw
            a = np.abs(s)
            mx2 = a.max()
            a = a / mx2 if mx2 > 0 else a
            return s.astype(np.float32), a.astype(np.float32)
        finally:
            self.model.train(was)

    def generate_batch(
        self, X: torch.Tensor, branch_idx: int,
        smooth_steps: int = 0, noise_sigma: float = 0.1,
        desc: str = "GradCAM",
    ) -> Tuple[np.ndarray, np.ndarray]:
        sl, al = [], []
        for i in tqdm(range(len(X)), desc=desc, unit="win"):
            cs, ca = self.generate(X[i:i+1], branch_idx, smooth_steps, noise_sigma)
            sl.append(cs); al.append(ca)
        return np.stack(sl), np.stack(al)


class TemporalGradientExtractor:
    """
    시점별 E_t와 dY/dE_t 추출.

    v3: forward hook에서 clone() 반환 + retain_grad()
    → LeakyReLU(inplace=True) 이후에도 안전.
    """

    def __init__(self, model: nn.Module, device: torch.device, branch_idx: int):
        self.model = model
        self.device = device
        _, branches = get_branches(model)
        if branch_idx >= len(branches):
            raise ValueError(f"branch_idx={branch_idx} >= {len(branches)}")
        self.target_block = branches[branch_idx].network[-1]
        self._act: Optional[torch.Tensor] = None
        self._handle: Optional = None

    def _hook(self):
        if self._handle is not None:
            return
        parent = self

        def fwd(m, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            cloned = o.clone()
            cloned.retain_grad()
            parent._act = cloned
            return cloned

        self._handle = self.target_block.register_forward_hook(fwd)

    def cleanup(self):
        if self._handle is not None:
            try: self._handle.remove()
            except: pass
            self._handle = None

    def __del__(self):
        self.cleanup()

    def extract_single(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        assert x.dim() == 3 and x.size(0) == 1
        self._hook()
        x = x.to(self.device).requires_grad_(True)
        was = self.model.training
        self.model.eval()
        try:
            self._act = None
            self.model.zero_grad(set_to_none=True)
            y = self.model(x)
            if self._act is None:
                raise RuntimeError("Hook failed")
            y.squeeze().backward()
            if self._act.grad is None:
                raise RuntimeError("Gradient not computed")
            E = self._act.detach()[0].transpose(0, 1).cpu().numpy()
            G = self._act.grad.detach()[0].transpose(0, 1).cpu().numpy()
            T = x.size(1)
            if E.shape[0] != T:
                E = _interp_2d(E, T)
                G = _interp_2d(G, T)
            return E.astype(np.float64), G.astype(np.float64)
        finally:
            self.model.train(was)
            self._act = None

    def extract_batch(
        self, X: torch.Tensor, desc: str = "Gradients",
    ) -> Tuple[np.ndarray, np.ndarray]:
        El, Gl = [], []
        for i in tqdm(range(len(X)), desc=desc, unit="win"):
            E, G = self.extract_single(X[i:i+1])
            El.append(E); Gl.append(G)
        return np.stack(El), np.stack(Gl)


def _interp_2d(arr: np.ndarray, T: int) -> np.ndarray:
    T_src, C = arr.shape
    out = np.empty((T, C), dtype=arr.dtype)
    xs, xt = np.linspace(0, 1, T_src), np.linspace(0, 1, T)
    for c in range(C):
        out[:, c] = np.interp(xt, xs, arr[:, c])
    return out


def extract_embeddings(
    model: nn.Module, X: torch.Tensor, branch_idx: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Target branch timewise embedding. Returns (N, T, C)."""
    model.eval()
    dev = next(model.parameters()).device
    _, branches = get_branches(model)
    block = branches[branch_idx].network[-1]
    buf = {}

    def hook(m, inp, out):
        buf["o"] = (out[0] if isinstance(out, (tuple, list)) else out).detach()

    h = block.register_forward_hook(hook)
    parts = []
    try:
        with torch.no_grad():
            for s in range(0, len(X), batch_size):
                _ = model(X[s:s+batch_size].to(dev))
                parts.append(buf["o"].transpose(1, 2).cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(parts).astype(np.float64)