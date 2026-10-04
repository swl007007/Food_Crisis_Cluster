"""Stage1: direct per-H map learning on the pooled 2014-2022 F/S split (R24, R32-R38, R44-R46).

Adapted from ``FEWSNETGeoXGBExperiment/src/partition/partition_opt.py``
(``get_c_b``, ``scan``, ``swap_partition_polygon``, ``select_macro_children``)
and the shared-root recursion of ``transformation.py`` (``partition`` level loop,
support/eligibility, complete-route gate). Changes, all from the accepted
contract: single crisis-F1 scan column; exact R46 45-55% prefix rule with
canonical area-ID ties (replaces FLEX/optimize_size); fixed 1000 iterations,
last candidate; 3 synchronous induced-neighbour smoothing rounds; R28/R29
original-key support; quartet children continue the ROOT global once;
membership depth <= 4 and no path-round cap; no Stage2, donor or forced
connectivity.

Areas: the learned universe of an H is the Stage1 multi-outcome areas (each
has F and S keys). Singletons and zero-outcome areas are outside the map and
route to global (R36).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil

import numpy as np
import pandas as pd

from ipcch_geoxgb import metrics, projection
from ipcch_geoxgb.errors import TechnicalError
from ipcch_geoxgb.quartet import Quartet

# ----------------------------------------------------------- R46 candidate size


def size_bounds(n: int) -> tuple[int, int] | None:
    """Allowed prefix sizes [lo, hi] for 45%-55% of N groups (integer arithmetic)."""
    lo = max(1, -(-9 * n // 20))
    hi = min(n - 1, (11 * n) // 20)
    if n < 2 or lo > hi:
        return None
    return lo, hi


def choose_prefix(scores: np.ndarray, area_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Score-maximizing sorted prefix within the R46 range (positions into the input).

    Rank by descending score, exact ties by ascending canonical area ID. Among
    allowed m pick the largest prefix sum; ties prefer smaller |2m - N|, then
    smaller m. Returns (s0 positions, s1 positions, m) or None if infeasible.
    """
    scores = np.asarray(scores, dtype=np.float64)
    area_ids = np.asarray(area_ids, dtype=np.int64)
    n = len(scores)
    bounds = size_bounds(n)
    if bounds is None:
        return None
    if not np.isfinite(scores).all():
        raise TechnicalError("scan scores contain NaN/Inf")
    order = np.lexsort((area_ids, -scores))
    prefix = np.cumsum(scores[order])
    lo, hi = bounds
    best_m, best_value = None, None
    for m in range(lo, hi + 1):
        value = prefix[m - 1]
        if best_m is None or value > best_value or (
            value == best_value and (abs(2 * m - n), m) < (abs(2 * best_m - n), best_m)
        ):
            best_m, best_value = m, value
    return order[:best_m], order[best_m:], best_m


# ----------------------------------------------------------- R32 scan


@dataclass
class ScanResult:
    s0: np.ndarray  # positions into the node's group arrays
    s1: np.ndarray
    m_init: int
    m_final: int
    rho: float
    score: np.ndarray


def scan(Y: np.ndarray, A: np.ndarray, area_ids: np.ndarray, iterations: int = 1000) -> ScanResult | None:
    """Single-column crisis-F1 error scan; fixed iterations, last candidate.

    c = Y - A (normalized FP+FN mass); b = sum(c) * Y / sum(Y). Location score
    g = c*log(rho) + b*(1 - rho); rho = sum c[s0] / sum b[s0] (1 if the
    expectation is 0, and 0 is replaced by 1). rho is the scan multiplier, not
    a population share, and the score is not a p-value.
    """
    Y = np.asarray(Y, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    c = Y - A
    total = Y.sum()
    b = c.sum() * (Y / total) if total > 0 else np.zeros_like(Y)

    def update(s0):
        expected = b[s0].sum()
        rho = c[s0].sum() / expected if expected > 0 else 1.0
        return 1.0 if rho == 0 else float(rho)

    init = choose_prefix(np.divide(c, b, out=np.zeros_like(c), where=b > 0), area_ids)
    if init is None:
        return None
    rho = update(init[0])
    s0, s1, m = init
    g = np.zeros_like(c)
    for _ in range(iterations):
        g = c * np.log(rho) + b * (1.0 - rho)
        s0, s1, m = choose_prefix(g, area_ids)
        rho = update(s0)
    return ScanResult(s0=s0, s1=s1, m_init=init[2], m_final=m, rho=rho, score=g)


# ----------------------------------------------------------- R38 smoothing


def smooth(assignment: dict[int, int], neighbours: dict[int, list[int]], rounds: int = 3) -> dict[int, int]:
    """Synchronous neighbour-plus-self vote over the node's candidate areas.

    Only areas holding a candidate label vote. A label is switched to the other
    child only when its share of (valid neighbours + self) is strictly < 4/9.
    Isolated areas or areas without a labelled neighbour keep their label.
    """
    current = dict(assignment)
    for _ in range(rounds):
        nxt = dict(current)
        for area, label in current.items():
            votes = [current[n] for n in neighbours.get(area, ()) if n in current]
            if not votes:
                continue
            total = len(votes) + 1
            same = sum(1 for v in votes if v == label) + 1
            if 9 * same < 4 * total:
                nxt[area] = 1 - label
        current = nxt
    return current


def component_diagnostics(areas: np.ndarray, neighbours: dict[int, list[int]]) -> dict:
    """Connected components of a region on its member-induced shared-boundary graph."""
    members = set(int(a) for a in areas)
    seen, sizes = set(), []
    for start in sorted(members):
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            node = stack.pop()
            size += 1
            for n in neighbours.get(node, ()):
                if n in members and n not in seen:
                    seen.add(n)
                    stack.append(n)
        sizes.append(size)
    isolated = sum(1 for a in members if not any(n in members for n in neighbours.get(a, ())))
    return {"areas": len(members), "components": len(sizes), "largest_component": max(sizes, default=0),
            "isolated_areas": isolated}


# ----------------------------------------------------------- R28/R29 support


def support(frame: pd.DataFrame) -> dict:
    """Original-key support of a pool (one row = one area x target month at this H)."""
    crisis = int(frame["crisis_truth"].sum())
    return {
        "keys": int(len(frame)),
        "areas": int(frame["admin_code"].nunique()),
        "target_months": int(frame["target_ord"].nunique()),
        "crisis_keys": crisis,
        "noncrisis_keys": int(len(frame) - crisis),
    }


def meets(record: dict, floor: dict) -> bool:
    return all(record[k] >= v for k, v in floor.items())


# ----------------------------------------------------------- route selection


def select_route(truth: tuple, parent: tuple, child: tuple, eligible: tuple) -> dict:
    """Implement.md clarification 1: complete parent S keys; baseline parent/parent;
    try child/parent, parent/child, child/child in that order (skipping
    ineligible children); update only on a strictly larger exact F1; accept iff
    best - base > 0. Undefined parent -> no split; undefined combos never win."""
    t = np.concatenate(truth)
    base_counts = metrics.crisis_counts(t, np.concatenate(parent))
    base = metrics.exact_f1(base_counts)
    out = {"scores": {"parent_parent": None if base is None else str(base)}, "choice": (False, False)}
    if base is None:
        return {**out, "accepted": False, "reason": "parent_f1_undefined"}
    best = base
    for use0, use1 in ((True, False), (False, True), (True, True)):
        if (use0 and not eligible[0]) or (use1 and not eligible[1]):
            continue
        pred = np.concatenate((child[0] if use0 else parent[0], child[1] if use1 else parent[1]))
        value = metrics.exact_f1(metrics.crisis_counts(t, pred))
        out["scores"][f"{'child' if use0 else 'parent'}_{'child' if use1 else 'parent'}"] = (
            None if value is None else str(value)
        )
        if value is not None and value > best:
            best, out["choice"] = value, (use0, use1)
    accepted = best - base > Fraction(0)
    return {**out, "accepted": bool(accepted), "base_f1": str(base), "best_f1": str(best),
            "reason": "gain_above_zero" if accepted else "no_strict_gain"}


# ----------------------------------------------------------- recursion


#: Root membership node. IDs are "r" + one binary digit per accepted split
#: ("r0", "r01", ...), never purely numeric, so no CSV reader can strip a
#: leading zero; membership depth = len(node_id) - 1.
ROOT_ID = "r"


def node_depth(node_id: str) -> int:
    if not node_id.startswith(ROOT_ID) or set(node_id[1:]) - {"0", "1"}:
        raise TechnicalError(f"malformed membership node id {node_id!r}")
    return len(node_id) - 1


@dataclass
class Node:
    node_id: str
    areas: np.ndarray  # sorted canonical area IDs
    provider: str  # quartet identity digest used to predict this node
    provider_kind: str  # "root_global" or "local"


@dataclass
class SearchResult:
    terminal: list[Node]
    decisions: list[dict] = field(default_factory=list)
    fits: dict = field(default_factory=dict)


def route_predictions(quartet: Quartet, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = quartet.predict_raw(X)
    q_star, phase = projection.project_and_decode(raw)
    return raw, q_star, phase


def search_candidate(
    *,
    fit_keys: pd.DataFrame,
    val_keys: pd.DataFrame,
    X_fit: np.ndarray,
    X_val: np.ndarray,
    root: tuple[str, Quartet],
    fit_local,  # callable(child_areas, rows_mask) -> (digest, Quartet)
    neighbours: dict[int, list[int]],
    contract: dict,
) -> tuple[SearchResult, dict[str, Quartet]]:
    """One (H, G, L) candidate: R45 depth-4 recursion from the root over learned areas."""
    part = contract["partition"]
    fit_floor = contract["support"]["local_fit"]
    val_floor = contract["support"]["validation"]
    root_digest, root_quartet = root
    providers: dict[str, Quartet] = {root_digest: root_quartet}
    learned = np.array(sorted(set(fit_keys["admin_code"]) | set(val_keys["admin_code"])), dtype=np.int64)
    if set(fit_keys["admin_code"]) != set(val_keys["admin_code"]):
        raise TechnicalError("Stage1 learned areas must each carry F and S keys")
    fit_area = fit_keys["admin_code"].to_numpy()
    val_area = val_keys["admin_code"].to_numpy()

    result = SearchResult(terminal=[])
    frontier = [Node(ROOT_ID, learned, root_digest, "root_global")]
    for depth in range(part["max_member_depth"] + 1):
        next_frontier = []
        for node in frontier:
            if depth >= part["max_member_depth"]:
                result.terminal.append(node)
                continue
            decision, children = _attempt_split(
                node, fit_keys, val_keys, X_fit, X_val, fit_area, val_area, providers,
                fit_local, neighbours, fit_floor, val_floor, part,
            )
            result.decisions.append(decision)
            if children is None:
                result.terminal.append(node)
            else:
                next_frontier.extend(children)
        frontier = next_frontier
        if not frontier:
            break
    result.terminal.sort(key=lambda n: n.node_id)
    return result, providers


def _attempt_split(node, fit_keys, val_keys, X_fit, X_val, fit_area, val_area, providers,
                   fit_local, neighbours, fit_floor, val_floor, part):
    record = {"node_id": node.node_id, "depth": node_depth(node.node_id), "areas": int(len(node.areas)),
              "provider": node.provider, "provider_kind": node.provider_kind}
    in_node = np.isin(val_area, node.areas)
    truth = val_keys["phase_truth"].to_numpy()[in_node]
    _, _, parent_phase = route_predictions(providers[node.provider], X_val[in_node])
    groups, Y, A, counts = metrics.crisis_scan_masses(truth, parent_phase, val_area[in_node])
    if not np.array_equal(groups, node.areas):
        raise TechnicalError(f"node {node.node_id!r}: S groups differ from node membership")
    exposure = int((2 * counts["tp"] + counts["fp"] + counts["fn"]).sum())
    errors = int((counts["fp"] + counts["fn"]).sum())
    record.update({"scan_groups": int(len(groups)), "exposure": exposure, "error_mass": errors})
    if exposure == 0 or errors == 0:
        return {**record, "outcome": "no_candidate_zero_error_mass"}, None
    found = scan(Y, A, groups, iterations=part["scan_iterations"])
    if found is None:
        return {**record, "outcome": "no_feasible_size", "size_bounds": None}, None
    record.update({"size_bounds": list(size_bounds(len(groups))), "m_init": found.m_init,
                   "m_final": found.m_final, "rho": found.rho})
    candidate = {int(a): 0 for a in groups[found.s0]}
    candidate.update({int(a): 1 for a in groups[found.s1]})
    smoothed = smooth(candidate, neighbours, part["smoothing_rounds"])
    side_areas = [np.array(sorted(a for a, v in smoothed.items() if v == k), dtype=np.int64) for k in (0, 1)]
    switched = sum(1 for a in candidate if candidate[a] != smoothed[a])
    record.update({"sizes_before_smoothing": [int(len(found.s0)), int(len(found.s1))],
                   "sizes_after_smoothing": [int(len(side_areas[0])), int(len(side_areas[1]))],
                   "smoothing_switched": switched})
    if any(len(s) == 0 for s in side_areas):
        return {**record, "outcome": "rejected_empty_child"}, None

    fit_mask = [np.isin(fit_area, s) for s in side_areas]
    val_mask = [np.isin(val_area, s) for s in side_areas]
    fit_sup = [support(fit_keys[m]) for m in fit_mask]
    val_sup = [support(val_keys[m]) for m in val_mask]
    eligible = tuple(meets(fit_sup[k], fit_floor) and meets(val_sup[k], val_floor) for k in (0, 1))
    record.update({"fit_support": fit_sup, "val_support": val_sup, "eligible": list(eligible)})
    if not any(eligible):
        return {**record, "outcome": "rejected_no_eligible_child"}, None

    child_quartets = [None, None]
    child_phase = [None, None]
    parent_phase_side = []
    truth_side = []
    for k in (0, 1):
        rows = val_mask[k]
        truth_side.append(val_keys["phase_truth"].to_numpy()[rows])
        parent_phase_side.append(route_predictions(providers[node.provider], X_val[rows])[2])
        if eligible[k]:
            digest, quartet = fit_local(side_areas[k], fit_mask[k])
            providers[digest] = quartet
            child_quartets[k] = digest
            child_phase[k] = route_predictions(quartet, X_val[rows])[2]
    gate = select_route(tuple(truth_side), tuple(parent_phase_side), tuple(child_phase), eligible)
    record.update({"gate": {k: v for k, v in gate.items() if k != "choice"},
                   "selected_children": list(gate["choice"]), "child_providers": child_quartets})
    if not gate["accepted"]:
        return {**record, "outcome": "rejected_gate" if gate["reason"] != "parent_f1_undefined"
                else "rejected_undefined_parent_metric"}, None
    children = []
    for k in (0, 1):
        use_child = gate["choice"][k]
        children.append(Node(
            node_id=node.node_id + str(k),
            areas=side_areas[k],
            provider=child_quartets[k] if use_child else node.provider,
            provider_kind="local" if use_child else node.provider_kind,
        ))
    return {**record, "outcome": "accepted"}, children


def terminal_routing(terminal: list[Node], area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(node_id, provider digest) per row; every row's area must be in a terminal region."""
    node_of, provider_of = {}, {}
    for node in terminal:
        for a in node.areas:
            node_of[int(a)] = node.node_id
            provider_of[int(a)] = node.provider
    node_ids = np.array([node_of.get(int(a)) for a in area], dtype=object)
    if any(v is None for v in node_ids):
        raise TechnicalError("an S key's area is not in any terminal region")
    return node_ids, np.array([provider_of[int(a)] for a in area], dtype=object)


# ----------------------------------------------------------- R44 selection


def selection_key(entry: dict) -> tuple:
    """Sort key: higher exact F1 first, then the R44 tie order."""
    return (
        -entry["f1_exact"],
        entry["terminal_regions"],
        entry["global_rounds"] + entry["local_rounds"],
        entry["global_depth"],
        entry["local_depth"],
        entry["g_id"],
        entry["l_id"],
    )


def select_winner(entries: list[dict]) -> dict:
    """R44: absolute S crisis F1; NA never wins; all-NA -> selection_unavailable."""
    defined = [e for e in entries if e["f1_exact"] is not None]
    if not defined:
        return {"status": "selection_unavailable", "winner": None, "ranking": []}
    ranked = sorted(defined, key=selection_key)
    return {
        "status": "selected",
        "winner": ranked[0]["candidate"],
        "ranking": [e["candidate"] for e in ranked],
        "undefined": [e["candidate"] for e in entries if e["f1_exact"] is None],
    }
