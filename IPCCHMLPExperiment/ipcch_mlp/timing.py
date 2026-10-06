"""P0 synthetic timing and determinism probe (design section 9). Synthetic data only.

For each device: G1/G2 global (100 epochs) and R1/R2 residual (40 epochs) at
n = 500/2000/8561/19052, one warmup and three measured repeats. Eligibility:
identical final tensors across repeats, identical predictions after save/load,
and identical retraining after an intervening fit (RNG isolation). The full-run
estimate interpolates fit time linearly in n over the enumerated pool sizes.
Device choice: the faster eligible device; ties choose CPU.
"""

from __future__ import annotations

import io
import time

import numpy as np

from ipcch_mlp import nets, runtime
from ipcch_mlp.runtime import torch

GRID = (500, 2000, 8561, 19052)
SEED = 20261005


def _data(n: int):
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n, nets.N_INPUTS)).astype(np.float32)
    X[:, 561:] = (rng.random((n, 561)) < 0.4).astype(np.float32)  # missingness-flag block
    y = (0.3 + 0.05 * X[:, 0] - 0.03 * X[:, 1] + rng.normal(0, 0.02, n)).clip(0, 1).astype(np.float32)
    return X, y


def _fit(widths, role, X, y, epochs, cfg, device):
    net = nets.build(widths, role, 11)
    t0 = time.perf_counter()
    h = nets.train(net, X, y, epochs, {"train": 12, "perm": 13}, cfg, device)
    if device == "cuda":
        torch.cuda.synchronize()
    return net, time.perf_counter() - t0, h


def probe_device(device: str, contract: dict, threads: int, grid=GRID, repeats: int = 3) -> dict:
    runtime.configure(device, threads)
    cfg = contract["training"]
    arch = contract["architecture"]
    cases = [("global", gid, arch["global_candidates"][gid], cfg["global_epochs"]) for gid in ("G1", "G2")] + \
            [("residual", rid, arch["residual_candidates"][rid], cfg["residual_epochs"]) for rid in ("R1", "R2")]
    out = {"device": device, "cases": [], "eligible": True, "problems": []}
    for n in grid:
        X, y = _data(n)
        for role, cid, widths, epochs in cases:
            _fit(widths, role, X, y, epochs, cfg, device)  # warmup
            times, digests = [], []
            for _ in range(repeats):
                net, dt, h = _fit(widths, role, X, y, epochs, cfg, device)
                times.append(dt)
                digests.append(nets.state_digest(nets.cpu_state(net)))
            t0 = time.perf_counter()
            pred = nets.predict(net, X, device, cfg["inference_batch_size"])
            t_inf = time.perf_counter() - t0
            buf = io.BytesIO()
            t0 = time.perf_counter()
            torch.save(nets.cpu_state(net), buf)
            t_save = time.perf_counter() - t0
            buf.seek(0)
            reloaded = nets.load(widths, torch.load(buf, map_location="cpu", weights_only=True))
            same_reload = np.array_equal(pred, nets.predict(reloaded, X, device, cfg["inference_batch_size"]))
            ok = len(set(digests)) == 1 and same_reload
            if not ok:
                out["eligible"] = False
                out["problems"].append(f"{role}/{cid}/n={n}: repeat_identical={len(set(digests)) == 1} reload={same_reload}")
            out["cases"].append({"n": n, "role": role, "candidate": cid, "epochs": epochs, "updates": h["updates"],
                                 "fit_seconds": times, "inference_seconds": t_inf, "save_seconds": t_save,
                                 "state_bytes": buf.getbuffer().nbytes, "repeat_identical": len(set(digests)) == 1,
                                 "reload_identical": same_reload})
    # RNG isolation: same fit, then an unrelated fit, then the same fit again
    X, y = _data(grid[0])
    a, _, _ = _fit([16], "residual", X, y, 5, cfg, device)
    _fit([64, 32], "global", X, y, 3, cfg, device)
    b, _, _ = _fit([16], "residual", X, y, 5, cfg, device)
    iso = nets.state_digest(nets.cpu_state(a)) == nets.state_digest(nets.cpu_state(b))
    out["rng_isolation_identical"] = iso
    if not iso:
        out["eligible"] = False
        out["problems"].append("RNG isolation failed")
    return out


def interpolate(cases: list, role: str, cid: str, n: int) -> tuple[float, float]:
    """Linear interpolation of (min, max) repeat fit seconds; n must lie inside the grid."""
    pts = sorted((c["n"], min(c["fit_seconds"]), max(c["fit_seconds"])) for c in cases
                 if c["role"] == role and c["candidate"] == cid)
    xs = [p[0] for p in pts]
    if not xs[0] <= n <= xs[-1]:
        raise ValueError(f"pool size {n} outside the timing grid {xs}")
    return float(np.interp(n, xs, [p[1] for p in pts])), float(np.interp(n, xs, [p[2] for p in pts]))


def estimate(cases: list, enumeration: dict, dev_fit_rows: int, replicates: int = 3) -> dict:
    """Seconds for the whole planned workload, per recipe, from min/max repeat timings."""
    out = {}
    for gid in ("G1", "G2"):
        for rid in ("R1", "R2"):
            lo = hi = 0.0
            for h, e in enumeration.items():
                for n in e["global_pool_sizes"]:
                    for role, cid in (("global", gid), ("residual", rid)):
                        a, b = interpolate(cases, role, cid, n)
                        lo, hi = lo + 4 * a, hi + 4 * b
                for n in e["regional_pool_sizes"]:
                    a, b = interpolate(cases, "residual", rid, n)
                    lo, hi = lo + 4 * a, hi + 4 * b
            out[gid + rid] = {"stage3_seconds_three_seeds": [replicates * lo, replicates * hi]}
    dev_lo = dev_hi = 0.0
    for gid in ("G1", "G2"):
        a, b = interpolate(cases, "global", gid, dev_fit_rows)
        dev_lo, dev_hi = dev_lo + a, dev_hi + b
        for rid in ("R1", "R2"):
            a, b = interpolate(cases, "residual", rid, dev_fit_rows)
            dev_lo, dev_hi = dev_lo + a, dev_hi + b
    out["develop_seconds"] = [4 * 4 * replicates * dev_lo, 4 * 4 * replicates * dev_hi]
    return out


def choose_device(results: dict, estimates: dict) -> str:
    eligible = {d: r for d, r in results.items() if r["eligible"]}
    if not eligible:
        return ""
    total = {d: max(v["stage3_seconds_three_seeds"][1] for k, v in estimates[d].items() if k != "develop_seconds")
             for d in eligible}
    best = min(total.values())
    return "cpu" if "cpu" in total and total["cpu"] <= best else min(total, key=total.get)
