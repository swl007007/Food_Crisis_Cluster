"""P1: development fits on original F, recipe selection on complete common S (PRD R14-R15, R20).

Per H and replicate: two B quartets (G1, G2) on F, each reused by the R1 and R2
pooled residual quartets (288 scalar fits over 4 H x 3 seeds). Each candidate is
scored by the projected/decoded crisis F1 of B+P on all S keys from merged
TP/FP/FN. One recipe per H: highest arithmetic mean of the three exact per-seed
F1 fractions; a candidate with any undefined seed F1 is ineligible; ties go to
fewer B+P parameters, then the smaller candidate ID. No refit on F+S and no
development regional networks.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_mlp import metrics, projection, sources
from ipcch_mlp.artifacts import sha256_file, write_json
from ipcch_mlp.quartets import TARGETS, Engine

GZ = {"method": "gzip", "mtime": 0}
CANDIDATES = ("G1R1", "G1R2", "G2R1", "G2R2")


def candidate_params(contract: dict, cid: str) -> int:
    pc = contract["architecture"]["parameter_counts"]
    return 4 * (pc[cid[:2]] + pc[cid[2:]])  # per quartet; the order equals the per-network order


def select(entries: dict, contract: dict) -> dict:
    """entries: cid -> {replicate: f1 str|None}. Exact mean selection with the fixed tie order."""
    ranked, ineligible = [], {}
    for cid in CANDIDATES:
        f1s = entries[cid]
        if any(v is None for v in f1s.values()):
            ineligible[cid] = "undefined_seed_f1"
            continue
        mean = sum((Fraction(v) for v in f1s.values()), Fraction(0)) / len(f1s)
        ranked.append((-mean, candidate_params(contract, cid), cid, mean))
    if not ranked:
        return {"status": "selection_unavailable", "winner": None, "ineligible": ineligible}
    ranked.sort(key=lambda r: (r[0], r[1], r[2]))
    return {"status": "selected", "winner": ranked[0][2],
            "ranking": [{"candidate": r[2], "mean_f1": str(r[3]), "mean_f1_float": float(r[3]), "params_quartet": r[1]}
                        for r in ranked],
            "ineligible": ineligible}


def score_rows(truth: np.ndarray, raw: np.ndarray) -> dict:
    star, phase = projection.project_and_decode(raw)
    counts = metrics.crisis_counts(truth, phase)
    f1 = metrics.exact_f1(counts)
    return {"star": star, "phase": phase, "counts": counts, "f1": None if f1 is None else str(f1)}


def run_develop(run_dir: Path, engine: Engine, contract: dict, horizons: dict, split: pd.DataFrame) -> dict:
    out = run_dir / "develop"
    out.mkdir()
    reps = contract["replicates"]
    summary = {"stage": "P1-develop", "horizons": {}}
    winners = {}
    for h, hz in horizons.items():
        roles = sources.split_roles(hz, split)
        F = np.flatnonzero(roles == "fit")
        S = np.flatnonzero(roles == "validation")
        exp = contract["development"]
        if (len(F), len(S)) != (exp["fit_keys"], exp["validation_keys"]):
            raise ValueError(f"H{h}: F/S = {len(F)}/{len(S)}, expected {exp['fit_keys']}/{exp['validation_keys']}")
        truth = hz.keys["phase_truth"].to_numpy()[S]
        hdir = out / f"h{h:02d}"
        hdir.mkdir()
        entries = {cid: {} for cid in CANDIDATES}
        records = []
        for rep in reps:
            for gid in ("G1", "G2"):
                use = {"stage": "develop", "H": h, "replicate": rep, "candidate_global": gid}
                gf = engine.global_fit(hz, "develop", rep, gid, "dev-F", F, use)
                XS = engine.inputs(hz, gf, S)
                B_S = gf.B.predict(XS, engine.device, engine.train_cfg["inference_batch_size"])
                for rid in ("R1", "R2"):
                    cid = gid + rid
                    Pq = engine.residual_fit(hz, gf, rid, None, {**use, "candidate": cid})
                    P_S = Pq.predict(XS, engine.device, engine.train_cfg["inference_batch_size"])
                    sc = score_rows(truth, B_S + P_S)
                    entries[cid][rep] = sc["f1"]
                    frame = pd.DataFrame({"row": S, "admin_code": hz.area[S], "target_ord": hz.keys["target_ord"].to_numpy()[S],
                                          "phase_truth": truth})
                    for j, q in enumerate(TARGETS):
                        frame[f"base_{q}"] = B_S[:, j]
                        frame[f"pres_{q}"] = P_S[:, j]
                        frame[f"pool_{q}_star"] = sc["star"][:, j]
                    frame["pool_phase"] = sc["phase"]
                    path = hdir / f"S_rep{rep}_{cid}.csv.gz"
                    frame.to_csv(path, index=False, compression=GZ)
                    records.append({"replicate": rep, "candidate": cid, "f1": sc["f1"], "counts": sc["counts"],
                                    "base_provider": gf.B.provider(), "pres_provider": Pq.provider(),
                                    "transform_sha256": gf.transform_sha256, "predictions_sha256": sha256_file(path),
                                    "global_updates": {q: gf.B.records[q]["history"]["updates"] for q in TARGETS},
                                    "residual_updates": {q: Pq.records[q]["history"]["updates"] for q in TARGETS},
                                    "global_final_loss": {q: gf.B.records[q]["history"]["epoch_loss"][-1] for q in TARGETS},
                                    "residual_final_loss": {q: Pq.records[q]["history"]["epoch_loss"][-1] for q in TARGETS}})
        decision = select(entries, contract)
        if decision["status"] != "selected":
            write_json(hdir / "selection.json", {"entries": entries, "selection": decision, "records": records})
            raise ValueError(f"H{h}: selection_unavailable")
        winners[str(h)] = decision["winner"]
        write_json(hdir / "selection.json", {"entries": entries, "selection": decision, "records": records,
                                             "note": "S is adaptive internal development evidence, not independent validation"})
        summary["horizons"][str(h)] = {"winner": decision["winner"], "entries": entries}
    write_json(out / "frozen_recipes.json", {"winners": winners, "contract": contract["contract_version"]})
    summary["store_counts"] = dict(engine.store.counts)
    write_json(out / "develop-summary.json", summary)
    return summary


def load_winners(run_dir: Path) -> dict:
    return json.loads((run_dir / "develop" / "frozen_recipes.json").read_text(encoding="utf-8"))["winners"]
