"""Scalar MLPs: construction, deterministic training and inference (design section 5).

Hidden block = Linear -> ReLU -> Dropout(0.10); linear scalar output. PyTorch
default initialization, except that the output layer of every pooled/regional
residual network is set to exactly zero (R16). Networks are constructed on CPU
under the identity-derived init seed, so initial weights do not depend on the
device. Training: float32 MSE, AdamW(lr .001, betas (.9,.999), eps 1e-8,
decay .01 on weight matrices only, amsgrad/foreach/fused False), batch
min(256, n), every row once per epoch incl. the last partial batch, a
dedicated CPU generator for epoch permutations, and a global reseed with the
training seed right before training (dropout masks use the global stream).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ipcch_mlp.errors import TechnicalError
from ipcch_mlp.runtime import torch

nn = torch.nn
N_INPUTS = 1122
DROPOUT = 0.10


class ScalarMLP(nn.Module):
    def __init__(self, widths: list[int], n_inputs: int = N_INPUTS, dropout: float = DROPOUT):
        super().__init__()
        layers: list = []
        prev = n_inputs
        for w in widths:  # forward order construction fixes the init draw order
            layers += [nn.Linear(prev, int(w)), nn.ReLU(), nn.Dropout(dropout)]
            prev = int(w)
        layers.append(nn.Linear(prev, 1))
        self.body = nn.Sequential(*layers)
        self.widths = [int(w) for w in widths]

    def forward(self, x):
        return self.body(x).squeeze(1)

    @property
    def output_layer(self):
        return self.body[-1]


def parameter_count(widths: list[int], n_inputs: int = N_INPUTS) -> int:
    total, prev = 0, n_inputs
    for w in list(widths) + [1]:
        total += prev * w + w
        prev = w
    return total


def build(widths: list[int], role: str, init_seed: int) -> ScalarMLP:
    if role not in ("global", "residual"):
        raise TechnicalError(f"unknown network role {role!r}")
    torch.manual_seed(init_seed)  # CPU and all CUDA generators
    net = ScalarMLP(widths)
    if role == "residual":
        with torch.no_grad():
            net.output_layer.weight.zero_()
            net.output_layer.bias.zero_()
    return net


def state_digest(state: dict) -> str:
    """Canonical tensor digest: sorted names, dtype, shape and C-contiguous CPU bytes."""
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().to("cpu").contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def cpu_state(net: ScalarMLP) -> dict:
    return {k: v.detach().to("cpu").clone() for k, v in net.state_dict().items()}


def _optimizer(net: ScalarMLP, cfg: dict):
    decay = [p for _, p in net.named_parameters() if p.ndim == 2]
    no_decay = [p for _, p in net.named_parameters() if p.ndim != 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"], amsgrad=False, foreach=False, fused=False,
    )


def train(net: ScalarMLP, X: np.ndarray, y: np.ndarray, epochs: int, seeds: dict, cfg: dict, device: str) -> dict:
    """Fixed-epoch training; returns the per-epoch row-weighted loss and update count."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    n = len(y)
    if n == 0 or X.shape != (n, N_INPUTS):
        raise TechnicalError(f"training data shape {X.shape} / {y.shape}")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise TechnicalError("training data are not finite")
    batch = min(int(cfg["batch_size"]), n)
    net = net.to(device)
    Xd = torch.from_numpy(X).to(device)
    yd = torch.from_numpy(y).to(device)
    opt = _optimizer(net, cfg)
    loss_fn = nn.MSELoss(reduction="mean")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seeds["perm"]))
    torch.manual_seed(int(seeds["train"]))
    losses = []
    for _ in range(int(epochs)):
        net.train()
        perm = torch.randperm(n, generator=gen).to(device)
        total = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            out = net(Xd[idx])
            loss = loss_fn(out, yd[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach().to(torch.float64) * len(idx)
        value = float(total.item()) / n
        if not math.isfinite(value):
            raise TechnicalError("training loss became non-finite")
        losses.append(value)
    net.eval()
    return {"epoch_loss": losses, "updates": int(epochs) * math.ceil(n / batch), "batch_size": batch, "n_fit": n}


def predict(net: ScalarMLP, X: np.ndarray, device: str, batch: int = 256) -> np.ndarray:
    """Evaluation mode, no gradients, fixed batches in saved row order; float64 output."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != N_INPUTS:
        raise TechnicalError(f"prediction input shape {X.shape}")
    if len(X) == 0:
        return np.zeros(0, dtype=np.float64)
    net = net.to(device)
    net.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            xb = torch.from_numpy(X[start:start + batch]).to(device)
            parts.append(net(xb).to("cpu"))
    out = torch.cat(parts).numpy().astype(np.float64)
    if out.shape != (len(X),) or not np.isfinite(out).all():
        raise TechnicalError("network prediction is non-finite or misshaped")
    return out


def load(widths: list[int], state: dict) -> ScalarMLP:
    net = ScalarMLP(widths)
    net.load_state_dict(state)
    net.eval()
    return net
