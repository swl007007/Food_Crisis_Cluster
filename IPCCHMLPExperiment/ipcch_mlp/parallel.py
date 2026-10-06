"""Replicate-parallel execution (design section 6, approved 2026-10-05).

The three replicates run as concurrent spawned worker processes, one replicate
per process. Inside a process every fit is still strictly serial, seeds and
RNG resets are per fit and derived from the fit identity, and replicates never
share a model identity, so fitted tensors and outputs equal the serial mode
(tested). Each worker writes its own request ledger
``model_requests_<stage>_rep<r>.jsonl``; model entries are written atomically
and transforms (identical across replicates for the same pool) are written by
atomic replace with identical bytes. Selection (develop) and the Stage3
summary are finalized by the parent only after every worker succeeded. A
failed worker leaves INCOMPLETE evidence and the run stops; nothing is retried.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
from pathlib import Path

from ipcch_mlp.artifacts import record_incomplete
from ipcch_mlp.errors import TechnicalError

WORKER_CACHE = 6


def resolve(spec: str):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def _worker(stage: str, run_dir: str, rep: int, loader: str, extra: dict) -> None:
    from ipcch_mlp import develop, stage3  # noqa: PLC0415
    from ipcch_mlp.quartets import Engine  # noqa: PLC0415
    from ipcch_mlp.store import ModelStore  # noqa: PLC0415

    run = Path(run_dir)
    try:
        ctx = resolve(loader)()
        store = ModelStore(run / "models", run / f"model_requests_{stage}_rep{rep}.jsonl")
        engine = Engine(store, ctx["contract"], ctx["env"], ctx["device"], cache_size=WORKER_CACHE)
        if stage == "develop":
            develop.develop_replicate(run, engine, ctx["contract"], ctx["horizons"], ctx["split"], rep)
        elif stage == "stage3":
            stage3.run_replicate(run, engine, ctx["contract"], ctx["horizons"], ctx["calendar"], extra["winners"], rep)
        else:
            raise ValueError(f"unknown stage {stage}")
    except BaseException as error:
        record_incomplete(run, f"{stage}_rep{rep}", {"stage": stage, "replicate": rep}, error)
        raise


def run_replicates(stage: str, run_dir: Path, replicates: list, loader: str, extra: dict | None = None) -> None:
    ctx = mp.get_context("spawn")
    procs = {rep: ctx.Process(target=_worker, args=(stage, str(run_dir), rep, loader, extra or {}),
                              name=f"{stage}-rep{rep}") for rep in replicates}
    for p in procs.values():
        p.start()
    for p in procs.values():
        p.join()
    failed = {rep: p.exitcode for rep, p in procs.items() if p.exitcode != 0}
    if failed:
        raise TechnicalError(f"{stage}: replicate workers failed (exit codes {failed}); see INCOMPLETE records")


def run_develop_parallel(run_dir: Path, contract: dict, loader: str) -> dict:
    from ipcch_mlp import develop  # noqa: PLC0415
    (run_dir / "develop").mkdir()
    run_replicates("develop", run_dir, contract["replicates"], loader)
    return develop.finalize_develop(run_dir, contract)


def run_stage3_parallel(run_dir: Path, contract: dict, loader: str, winners: dict) -> dict:
    from ipcch_mlp import stage3  # noqa: PLC0415
    (run_dir / "stage3").mkdir()
    run_replicates("stage3", run_dir, contract["replicates"], loader, {"winners": winners})
    return stage3.finalize_stage3(run_dir, contract, winners)
