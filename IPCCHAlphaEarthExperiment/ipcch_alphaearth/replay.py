"""Independent zero-XGB-fit replay (design section 13).

Read-only model store (any fit request raises). Checks, each recorded with pass/fail and counts:

1. backbone: every stored fit identity recomputed from the inputs (keys/X/y/weights/params/role/origin) equals its
   digest; every saved provider/development/main/supplement raw prediction re-predicted bit-for-bit.
2. providers/pools: the complete expected provider table (every key inside a B1 block range, with that block's
   actual request identity, fit origin and re-predicted raw quartet) equals the saved historical and combined
   tables; every saved correction pool equals the pool rebuilt from U <= O, including provider identities.
   Saved canonical fields (admin, country, target/origin, H/period/block, truth, q targets, persistence) are
   rebuilt from the inputs and the frozen schedule; no output authorizes its own cohort.
3. development: transforms/B2 re-solved; every accepted split re-derived from the saved candidate ledger (strict
   floor + tie order) and its joint objective re-solved from row statistics; recipe G phases re-predicted from
   saved coefficients; gains/guards recomputed and the D13 selection reproduced.
4. maps: the final map relearned with the selected recipe equals the frozen map (thresholds by hex) and membership.
5. predict: annual transforms/B2/G coefficients re-solved and support re-derived; B2/G raw re-predicted from the
   saved coefficients; projection checked against sklearn isotonic regression and decoding recounted.
6. report: every published panel (main, annual, E_persist incl. persistence, supplement, coverage subsets), contrast,
   annual sign, guard, status, qualifying list and country diagnostic recounted by an independent implementation
   from canonical truth; undefined scores compare as undefined with a reason. Bootstrap country counts, seeded
   multiplicities, per-draw F1/deltas and the linear 1.25th percentile are regenerated and compared to the
   retained draws.

Limitations (disclosed): production replay re-solves only accepted split states, not every rejected candidate;
it never refetches raw EE pixels (descriptor replay is offline reaggregation from saved moments when present).
"""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ipcch_alphaearth import backbone, correction as cr, evaluation as ev, partition as pt, projection, replay_report as rr, schedule
from ipcch_alphaearth.artifacts import digest, read_json, record_incomplete, sha256_file, write_json
from ipcch_alphaearth.errors import ContractError
from ipcch_alphaearth.pipeline import TARGETS, Inputs, Workspace, correction_pool, make_pool, recipe_id, verify_map_freeze
from ipcch_alphaearth.report import frame_panel

REL_TOL = 1e-9


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name: str, ok: bool, **detail):
        self.items.append({"check": name, "pass": bool(ok), **detail})

    @property
    def passed(self) -> bool:
        return all(i["pass"] for i in self.items)


def _close(a, b, rel=REL_TOL) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= rel * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))))


def run_replay(inputs: Inputs, contract: dict, run_dir: Path, store_root: Path) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / "replay"
    if (out / "COMPLETE.json").exists():
        raise ContractError("replay is COMPLETE")
    for stage in ("develop", "maps", "predict", "report"):
        if not (run_dir / stage / "COMPLETE.json").exists():
            raise ContractError(f"replay needs a COMPLETE {stage} stage")
    out.mkdir(exist_ok=True)
    try:
        return _replay(inputs, contract, run_dir, out, store_root)
    except Exception as error:
        record_incomplete(run_dir, "replay", {"stage": "replay"}, error)
        raise


def _replay(inputs: Inputs, contract: dict, run_dir: Path, out: Path, store_root: Path) -> dict:
    t0 = time.time()
    store = backbone.ModelStore(store_root, read_only=True)
    checks = Checks()
    verify_map_freeze(Workspace(run_dir), sorted(inputs.horizons))
    checks.add("map_freeze_hash_chain", True)
    c = contract["correction"]
    lam_s, tol = float(c["shared_lambda"]), float(c["solver_residual_tolerance"])
    verdicts = {}
    for h, hi in sorted(inputs.horizons.items()):
        blocks = {b.block_id: b for b in schedule.horizon_blocks(contract, h)}
        ledgers = pd.concat([pd.read_csv(run_dir / s / "request_ledger.csv.gz") for s in ("develop", "predict")], ignore_index=True)
        ledgers = ledgers[ledgers["horizon"] == h]
        boosters, identity_of = {}, {}
        for row in ledgers.drop_duplicates(["arm", "block_id"]).itertuples(index=False):
            b = blocks[row.block_id]
            bst, rec = store.load_by_digest(row.identity)
            ident = rec["identity"]
            rows = np.flatnonzero(hi.target <= b.fit_origin)
            X = hi.X(row.arm)
            w = schedule.decay_weights(hi.target[rows], b.fit_origin, contract["calendar"]["half_life_months"])
            again = backbone.fit_identity(arm=row.arm, horizon=h, block_id=b.block_id, role=b.role, fit_origin=b.fit_origin, rows=rows,
                                          keys=np.column_stack([hi.admin[rows], hi.target[rows]]), X=np.ascontiguousarray(X[rows]),
                                          Y=hi.Y[rows], w=w, features=inputs.features424 if row.arm == "B1" else inputs.features296,
                                          contract=contract, env=ident["env"], source=ident["source"])
            checks.add(f"h{h:02d}:{row.arm}:{b.block_id}:identity_recomputed", digest(again) == row.identity)
            pred_rows = np.flatnonzero((hi.target >= b.target_lo) & (hi.target <= b.target_hi))
            boosters[(row.arm, b.block_id)] = (pred_rows, backbone.predict_quartet(bst, X[pred_rows], int(contract["backbone"]["fixed"]["n_jobs"])))
            identity_of[(row.arm, b.block_id)] = row.identity
        if len(boosters) != sum(len(b.arms) for b in blocks.values()):
            checks.add(f"h{h:02d}:all_scheduled_requests_present", False, found=len(boosters))
        else:
            checks.add(f"h{h:02d}:all_scheduled_requests_present", True, found=len(boosters))
        # ---------------- providers: complete expected tables from verified identities and re-predictions
        expected = {}
        for bid, b in sorted(blocks.items(), key=lambda kv: kv[1].target_lo):
            rows, raw = boosters[("B1", bid)]
            expected[bid] = pd.DataFrame({"row": rows, "target_ord": hi.target[rows], "provider_block": bid,
                                          "provider_fit_origin": b.fit_origin, "provider_identity": identity_of[("B1", bid)],
                                          **{f"p_{q}": raw[:, k] for k, q in enumerate(TARGETS)}})
        exp_all = pd.concat(expected.values(), ignore_index=True).sort_values("row", kind="mergesort").reset_index(drop=True)
        exp_hist = exp_all[exp_all["provider_block"].map(lambda x: blocks[x].role == "historical")].reset_index(drop=True)
        in_range = np.zeros(len(hi.target), dtype=bool)
        for b in blocks.values():
            in_range |= (hi.target >= b.target_lo) & (hi.target <= b.target_hi)
        checks.add(f"h{h:02d}:expected_provider_keys_complete", np.array_equal(exp_all["row"].to_numpy(), np.flatnonzero(in_range))
                   and not exp_all["row"].duplicated().any() and bool((exp_all["provider_fit_origin"] < exp_all["target_ord"]).all()))
        for label, path, exp in (("historical", run_dir / "develop" / f"providers_h{h:02d}.csv.gz", exp_hist),
                                 ("combined", run_dir / "predict" / f"providers_all_h{h:02d}.csv.gz", exp_all)):
            saved = pd.read_csv(path, float_precision="round_trip").sort_values("row", kind="mergesort").reset_index(drop=True)
            checks.add(f"h{h:02d}:{label}_providers_equal_expected", _frames_equal(saved, exp, list(exp.columns)))
        prov, hist_prov = exp_all, exp_hist
        # ---------------- development
        sel_saved = read_json(run_dir / "develop" / f"selection_h{h:02d}.json")
        dev_pred = pd.read_csv(run_dir / "develop" / f"development_predictions_h{h:02d}.csv.gz", float_precision="round_trip")
        records = {}
        for b in [b for b in blocks.values() if b.development_year is not None]:
            ctx = f"h{h:02d}_dev_{b.development_year}"
            saved = read_json(run_dir / "develop" / f"context_{ctx}.json")
            pool = correction_pool(hi, hist_prov, b.fit_origin, contract)
            saved_pool = pd.read_csv(run_dir / "develop" / f"pool_{ctx}.csv.gz", float_precision="round_trip")
            checks.add(f"{ctx}:pool_rebuilt", np.array_equal(saved_pool["row"].to_numpy(), pool["rows"])
                       and np.array_equal(saved_pool["weight"].to_numpy(), pool["w"])
                       and list(saved_pool["provider_block"]) == list(pool["provider_blocks"])
                       and list(saved_pool["provider_identity"]) == list(pool["provider_identities"]))
            rows, p1 = boosters[("B1", b.block_id)]
            _, p0 = boosters[("B0", b.block_id)]
            part = dev_pred[dev_pred["context"] == ctx]
            k = hi.keys.iloc[rows]
            checks.add(f"{ctx}:dev_canonical_keys", len(part) == len(rows)
                       and np.array_equal(part["admin_code"].to_numpy(), k["admin_code"].to_numpy())
                       and list(part["country_key"].astype(str)) == list(k["country_key"].astype(str))
                       and np.array_equal(part["target_ord"].to_numpy(), k["target_ord"].to_numpy())
                       and np.array_equal(part["phase_truth"].to_numpy(), k["phase_truth"].to_numpy()))
            checks.add(f"{ctx}:dev_backbone_bitwise", np.array_equal(part[[f"B1_{q}_raw" for q in TARGETS]].to_numpy(), p1)
                       and np.array_equal(part[[f"B0_{q}_raw" for q in TARGETS]].to_numpy(), p0))
            truth = hi.keys["phase_truth"].to_numpy()[rows]
            if len(pool["rows"]) == 0:
                checks.add(f"{ctx}:identity_empty_pool_declared", bool(saved.get("identity_empty_pool")) and not saved["lambdas"]
                           and np.array_equal(part[[f"B2_{q}_raw" for q in TARGETS]].to_numpy(), p1))
                b2p = _panel(hi, rows, p1)
                panels = {"B0": _panel(hi, rows, p0), "B1": b2p, "B2": b2p}
                guards = ev.guard_check(b2p, {"B1": panels["B1"], "B2": panels["B2"], "B0": panels["B0"]}, contract)["pass"]
                for lam in (float(x) for x in c["regional_lambdas"]):
                    for k in (int(x) for x in c["leaf_budgets"]):
                        rid = recipe_id(k, lam)
                        rec = records.setdefault(rid, {"recipe_id": rid, "leaf_budget": k, "lambda_region": lam, "years": {}})
                        rec["years"][b.development_year] = {"gain": ev.f1_gain(b2p, b2p), "guards_pass": guards, "n_leaves": 1, "depth": 0}
                continue
            transform, ppool = make_pool(hi, pool)
            b2 = cr.solve_b2(ppool.total, transform.active(), lam_s, tol)
            checks.add(f"{ctx}:transform_b2_resolved", transform.record() == saved["transform"] and _close(b2["beta"], saved["beta_b2"]))
            V_eval = transform.design(p1, hi.Z[rows])
            b2_eval = cr.correct(p1, V_eval, np.asarray(saved["beta_b2"]))
            panels = {"B0": _panel(hi, rows, p0), "B1": _panel(hi, rows, p1), "B2": _panel(hi, rows, b2_eval)}
            for lam_text, states in saved["lambdas"].items():
                cand = pd.read_csv(run_dir / "develop" / f"candidates_{ctx}_lam{lam_text}.csv.gz", float_precision="round_trip")
                checks.add(f"{ctx}:lam{lam_text}:accepted_splits_rederived", _accepted_ok(cand, states, contract))
                for st in states[1:]:
                    pmap = pt.PartitionMap.from_record(st["map"])
                    resolved = _resolve_state(hi, ppool, pool, transform, pmap, float(lam_text), contract)
                    checks.add(f"{ctx}:lam{lam_text}:step{st['step']}:objective_resolved",
                               _close(resolved["objective"], st["objective"]) and _close(resolved["beta"], st["fit"]["beta"]))
                for k in (int(x) for x in c["leaf_budgets"]):
                    rid = recipe_id(k, float(lam_text))
                    st = [s for s in states if s["map"]["n_leaves"] <= k][-1]
                    pmap = pt.PartitionMap.from_record(st["map"])
                    if st["fit"]["kind"] == "joint" and pmap.n_leaves >= 2:
                        fit = cr.CorrectionFit(transform, np.asarray(saved["beta_b2"]), saved["b2_objective"], "joint",
                                               np.asarray(st["fit"]["beta"]), st["fit"]["leaf_ids"], np.asarray(st["fit"]["u"]))
                    else:
                        fit = cr.CorrectionFit(transform, np.asarray(saved["beta_b2"]), saved["b2_objective"], "b2_reuse")
                    g = fit.predict(p1, hi.Z[rows], pmap.route(hi.D[rows], hi.routable[rows]))["g"]
                    _, gphase = projection.project_and_decode(g)
                    checks.add(f"{ctx}:{rid}:G_phase_repredicted", np.array_equal(gphase, part[f"G_{rid}_phase"].to_numpy()))
                    gp = _panel(hi, rows, g)
                    rec = records.setdefault(rid, {"recipe_id": rid, "leaf_budget": k, "lambda_region": float(lam_text), "years": {}})
                    fg, fb = _indep_f1(truth, gphase), _indep_f1(truth, projection.project_and_decode(b2_eval)[1])
                    rec["years"][b.development_year] = {"gain": None if fg is None or fb is None else fg - fb,
                                                        "guards_pass": ev.guard_check(gp, {"B1": panels["B1"], "B2": panels["B2"], "B0": panels["B0"]}, contract)["pass"],
                                                        "n_leaves": pmap.n_leaves, "depth": pmap.depth if pmap.n_leaves > 1 else 0}
        sel = ev.select_recipe(list(records.values()))
        checks.add(f"h{h:02d}:selection_reproduced", sel["selected"] == sel_saved["selected"], saved=sel_saved["selected"], replayed=sel["selected"])
        # ---------------- maps
        frozen = read_json(run_dir / "maps" / f"final_map_h{h:02d}.json")
        if frozen["map"] is not None:
            freeze = int(contract["calendar"]["map_freeze_cutoff"][:4]) * 12 + int(contract["calendar"]["map_freeze_cutoff"][5:]) - 1
            pool = correction_pool(hi, hist_prov, freeze, contract)
            transform, ppool = make_pool(hi, pool)
            search = pt.Search(ppool, inputs.D_admin, inputs.admin_codes, transform.active(), contract, float(frozen["lambda_region"]), inputs.descriptor_names)
            st = pt.state_at_budget(search.run(int(frozen["leaf_budget"]), int(c["max_depth"])), int(frozen["leaf_budget"]))
            checks.add(f"h{h:02d}:final_map_relearned", st["map"].record()["splits"] == frozen["map"]["splits"]
                       and sorted(st["map"].leaves) == frozen["map"]["leaves"])
            mem = pd.read_csv(run_dir / "maps" / f"membership_h{h:02d}.csv.gz")
            routable_admin = np.isfinite(inputs.D_admin).all(axis=1)
            again = st["map"].route(inputs.D_admin, routable_admin)
            checks.add(f"h{h:02d}:membership_rerouted", [None if pd.isna(x) else x for x in mem["leaf"]] == list(again))
        else:
            checks.add(f"h{h:02d}:final_map_b2_fallback_recorded", sel_saved["selected"] == "B2")
        # ---------------- predict
        # same NA policy as the canonical keys (runinputs): an empty source month stays "", empty numeric cells are NaN
        pred = pd.read_csv(run_dir / "predict" / f"predictions_h{h:02d}.csv.gz", float_precision="round_trip", keep_default_na=False,
                           na_values={c: [""] for c in ("q2", "q3", "q4", "q5", "persistence_q3")})
        contexts = read_json(run_dir / "predict" / f"contexts_h{h:02d}.json")
        pmap = None if frozen["map"] is None else pt.PartitionMap.from_record(frozen["map"])
        canonical_parts = []
        for cx in contexts:
            b = blocks[cx["block"]["block_id"]]
            prior = prov[prov["target_ord"] <= b.fit_origin]
            pool = correction_pool(hi, prior, b.fit_origin, contract)
            saved_pool = pd.read_csv(run_dir / "predict" / f"pool_{cx['context']}.csv.gz", float_precision="round_trip")
            checks.add(f"{cx['context']}:pool_rebuilt", np.array_equal(saved_pool["row"].to_numpy(), pool["rows"])
                       and np.array_equal(saved_pool["weight"].to_numpy(), pool["w"])
                       and list(saved_pool["provider_identity"]) == list(pool["provider_identities"]))
            saved_fit = cr.CorrectionFit.from_record(cx["correction"])
            if len(pool["rows"]) == 0:
                ok_coef = saved_fit.g_kind == "identity_empty_pool"
            else:
                transform, ppool = make_pool(hi, pool)
                b2 = cr.solve_b2(ppool.total, transform.active(), lam_s, tol)
                ok_coef = transform.record() == cx["correction"]["transform"] and _close(b2["beta"], saved_fit.beta_b2)
            if len(pool["rows"]) and pmap is not None and pmap.n_leaves >= 2:
                fr = pt.fit_frozen_map(ppool, transform.active(), pmap, pmap.route(hi.D[pool["rows"]], pool["routable"]), contract,
                                       float(frozen["lambda_region"]))
                ok_coef &= fr["g_kind"] == saved_fit.g_kind and fr["support"] == cx["support"]
                if fr["g_kind"] == "joint":
                    ok_coef &= _close(fr["beta_g"], saved_fit.beta_g) and _close(fr["u"], saved_fit.u)
            checks.add(f"{cx['context']}:annual_coefficients_resolved", bool(ok_coef))
            rows, p1 = boosters[("B1", b.block_id)]
            _, p0 = boosters[("B0", b.block_id)]
            part = pred[pred["block_id"] == b.block_id]
            leaf_of = pmap.route(hi.D[rows], hi.routable[rows]) if pmap is not None else np.full(len(rows), None, dtype=object)
            again = saved_fit.predict(p1, hi.Z[rows], leaf_of)
            ok = np.array_equal(part["row"].to_numpy(), rows)
            for arm, raw in (("B0", p0), ("B1", p1), ("B2", again["b2"]), ("G", again["g"])):
                ok &= np.array_equal(part[[f"{arm}_{q}_raw" for q in TARGETS]].to_numpy(), raw)
            ok &= list(part["G_route"]) == list(again["route"])
            checks.add(f"{cx['context']}:predictions_repredicted_bitwise", bool(ok))
            canon = rr.canonical_block(hi, b)
            checks.add(f"{cx['context']}:canonical_cohort_fields", _frames_equal(part.reset_index(drop=True), canon, rr.CANONICAL))
            verified = canon.copy()
            for arm, raw in (("B0", p0), ("B1", p1), ("B2", again["b2"]), ("G", again["g"])):
                star, phase = projection.project_and_decode(raw)
                for kq, q in enumerate(TARGETS):
                    verified[f"{arm}_{q}_raw"], verified[f"{arm}_{q}_star"] = raw[:, kq], star[:, kq]
                verified[f"{arm}_phase"] = phase
            verified["G_route"] = again["route"]
            canonical_parts.append(verified)
        expected_blocks = sorted(b.block_id for b in blocks.values() if b.role in ("main", "supplement"))
        checks.add(f"h{h:02d}:all_main_supplement_blocks_scored", sorted(set(pred["block_id"])) == expected_blocks
                   and sorted(cx["block"]["block_id"] for cx in contexts) == expected_blocks
                   and len(pred) == sum(len(x) for x in canonical_parts))
        checks.add(f"h{h:02d}:projected_columns_from_raw", rr.projected_columns_ok(pred))
        checks.add(f"h{h:02d}:projection_independent", _projection_ok(pred))
        canonical = pd.concat(canonical_parts, ignore_index=True)
        # ---------------- report: independent recount on the canonical cohort
        report_all = read_json(run_dir / "report" / "report.json")
        recount, recomputed_status = _recount_report(canonical, report_all["horizons"][str(h)], run_dir, h, contract)
        for name, ok in recount:
            checks.add(f"h{h:02d}:report:{name}", ok)
        verdicts[str(h)] = recomputed_status  # replay's own status, never the report's
    report_all = read_json(run_dir / "report" / "report.json")
    checks.add("report:qualifying_horizons", sorted(report_all["qualifying_horizons"]) ==
               sorted(h for h, st in verdicts.items() if st == "protected_stable_benefit"))
    fits = [r for r in store.ledger if r["action"] == "fit"]
    checks.add("zero_xgb_fits", len(fits) == 0, fits=len(fits))
    result = {"pass": checks.passed, "checks": checks.items, "n_checks": len(checks.items),
              "failed": [i["check"] for i in checks.items if not i["pass"]], "elapsed_seconds": round(time.time() - t0, 1),
              "limitations": ["rejected candidate splits are not individually re-solved; accepted states and ledger ordering are",
                              "raw EE pixels are never refetched"]}
    write_json(out / "replay.json", result)
    if not checks.passed:
        raise ContractError(f"replay failed {len(result['failed'])} checks: {result['failed'][:8]}")
    write_json(out / "COMPLETE.json", {"replay_sha256": sha256_file(out / "replay.json")})
    return result


def _recount_report(canonical: pd.DataFrame, rep: dict, run_dir: Path, h: int, contract: dict) -> list:
    """Every published number of one H recounted from the canonical cohort (independent code in replay_report)."""
    out = []
    main = canonical[canonical["period"] == "main"].reset_index(drop=True)
    supp = canonical[canonical["period"] == "supplement"].reset_index(drop=True)
    sc = {arm: rr.arm_scores(main, arm) for arm in rr.ARMS}
    out.append(("main_keys_countries", rep["main_keys"] == len(main) and rep["countries"] == main["country_key"].nunique()))
    out.append(("main_panels", all(rr.flat_ok(rep["panels"][a], sc[a]) and rr.detail_ok(rep["panel_details"][a], sc[a]) for a in rr.ARMS)))
    pairs = (("G", "B2"), ("G", "B1"), ("G", "B0"), ("B1", "B0"), ("B2", "B1"))
    out.append(("contrasts", all(rr.delta_ok(rep["contrasts"][f"{a}-{b}"], rr.delta(sc[a], sc[b])) for a, b in pairs)))
    years = {int(y): {arm: rr.arm_scores(part, arm) for arm in rr.ARMS} for y, part in main.groupby(main["target_ord"] // 12)}
    out.append(("annual_panels", sorted(rep["annual"]) == sorted(str(y) for y in years) and all(
        rr.flat_ok(rep["annual"][str(y)][a], ys[a]) and rr.detail_ok(rep["annual_details"][str(y)][a], ys[a])
        for y, ys in years.items() for a in rr.ARMS)))
    persist = main[main["persistence_available"] == 1]
    ps = {arm: rr.arm_scores(persist, arm) for arm in (*rr.ARMS, "persistence")}
    pr = rep["persistence"]
    out.append(("persistence_panels", pr["keys"] == len(persist)
                and rr._num_eq(pr["coverage"], len(persist) / len(main) if len(main) else None)
                and all(rr.flat_ok(pr["panels"][a], ps[a]) and rr.detail_ok(pr["panel_details"][a], ps[a]) for a in ps)
                and all(rr.delta_ok(pr["model_minus_persistence"][a], rr.delta(ps[a], ps["persistence"])) for a in rr.ARMS)))
    sp = rep["supplement_2026"]
    if len(supp):
        ss = {arm: rr.arm_scores(supp, arm) for arm in rr.ARMS}
        out.append(("supplement_panels", sp["keys"] == len(supp) and all(rr.flat_ok(sp["panels"][a], ss[a]) and
                                                                           rr.detail_ok(sp["panel_details"][a], ss[a]) for a in rr.ARMS)))
    else:
        out.append(("supplement_panels", sp["keys"] == 0 and sp["panels"] is None))
    subsets = {}
    for name, part in (("routable", main[main["routable"] == 1]), ("not_routable", main[main["routable"] == 0])):
        if len(part):
            subsets[name] = (len(part), rr.delta(rr.arm_scores(part, "G"), rr.arm_scores(part, "B2")))
    out.append(("coverage_subsets", set(rep["coverage_subsets"]) == set(subsets) and all(
        rep["coverage_subsets"][n]["keys"] == k and rr.delta_ok(rep["coverage_subsets"][n]["G-B2"], d) for n, (k, d) in subsets.items())))
    routes = main["G_route"].astype(str).str.split(":").str[0].value_counts().to_dict()
    out.append(("g_routes", rep["g_routes"] == routes))
    # decision
    gain = None if sc["G"]["binary.f1"] is None or sc["B2"]["binary.f1"] is None else sc["G"]["binary.f1"] - sc["B2"]["binary.f1"]
    annual = {}
    for y, ys in years.items():
        annual[y] = None if ys["G"]["binary.f1"] is None or ys["B2"]["binary.f1"] is None else ys["G"]["binary.f1"] - ys["B2"]["binary.f1"]
    gp, failed = rr.guards(sc["G"], sc["B1"], sc["B2"], sc["B0"], contract)
    boot = rr.bootstrap(main, contract)
    d = rep["decision"]
    out.append(("decision_gain_annual", d["f1_gain"] == (None if gain is None else str(gain)) and all(
        d["annual"][str(y)]["gain"] == (None if v is None else str(v)) and d["annual"][str(y)]["nonnegative"] == (None if v is None else v >= 0)
        for y, v in annual.items()) and sorted(d["annual"]) == sorted(str(y) for y in annual)))
    items = rr.guard_items(sc["G"], sc["B1"], sc["B2"], sc["B0"], contract)
    saved_items = {i["guard"]: i["pass"] for i in d["guards"]["items"]}
    out.append(("decision_guards", d["guards"]["pass"] == gp and sorted(d["guards"]["failed"]) == sorted(failed)
                and saved_items == items and len(d["guards"]["items"]) == len(items)))
    floor = Fraction(contract["evaluation"]["main_f1_gain_minimum"])
    c1 = gain is not None and gain >= floor
    c2 = boot["lower"] is not None and boot["lower"] > 0
    c3 = len(annual) > 0 and all(v is not None and v >= 0 for v in annual.values())
    out.append(("decision_criteria", d["criterion_gain_floor"] == c1 and d["criterion_corrected_lower_bound"] == c2
                and d["criterion_annual_signs"] == c3 and rr._num_eq(d["f1_gain_float"], None if gain is None else float(gain))
                and d["lower_bound"] == boot["lower"] and (d["bound_na_reason"] or "") == "; ".join(boot["reasons"])
                and all(rr._num_eq(d["annual"][str(y)]["gain_float"], None if v is None else float(v)) for y, v in annual.items())))
    recomputed_status = rr.status(gain, boot["lower"], annual, gp, contract)
    out.append(("decision_status", d["status"] == recomputed_status))
    # bootstrap evidence
    b = rep["bootstrap"]
    draws_path = run_dir / "report" / f"bootstrap_draws_h{h:02d}.csv.gz"
    ok = b["countries"] == boot["countries"] and b.get("lower_bound") == boot["lower"] and b.get("defined_draws") == boot["defined"]
    ok &= b.get("undefined_draws") == boot["undefined"] and b["K"] == boot["K"] and b["draws"] == boot["draws"] and b["seed"] == boot["seed"]
    ok &= float(b["percentile"]) == boot["percentile"] and (b.get("na_reason") or "") == "; ".join(boot["reasons"])
    ok &= b.get("point_delta_exact") == (None if boot["point"] is None else str(boot["point"]))
    ok &= rr._num_eq(b.get("point_delta"), None if boot["point"] is None else float(boot["point"]))
    ok &= (b.get("descriptive_uncorrected_95") is None and boot["descriptive"] is None) or (
        boot["descriptive"] is not None and all(rr._num_eq(x, y, 1e-15) for x, y in zip(b.get("descriptive_uncorrected_95") or [], boot["descriptive"])))
    ok &= b.get("lower_bound") == d["lower_bound"]
    ok &= b.get("country_counts", {}).get("G_phase") == boot["counts"]["G"].tolist() and b["country_counts"].get("B2_phase") == boot["counts"]["B2"].tolist()
    if draws_path.exists():
        saved = pd.read_csv(draws_path, float_precision="round_trip")
        ok &= np.array_equal(saved["delta"].to_numpy(), boot["delta"], equal_nan=True)
        ok &= np.array_equal(saved["f1_G_phase"].to_numpy(), boot["f1"]["G"], equal_nan=True)
        ok &= np.array_equal(saved["f1_B2_phase"].to_numpy(), boot["f1"]["B2"], equal_nan=True)
        ok &= np.array_equal(saved[[f"m::{cname}" for cname in boot["countries"]]].to_numpy(), boot["mult"])
    else:
        ok &= len(boot["countries"]) == 0
    out.append(("bootstrap_regenerated", bool(ok)))
    # country diagnostics
    diag = pd.read_csv(run_dir / "report" / f"country_diagnostics_h{h:02d}.csv.gz", float_precision="round_trip")
    good = sorted(diag["country_key"].astype(str)) == sorted(main["country_key"].astype(str).unique())
    for r in diag.itertuples(index=False):
        part = main[main["country_key"].astype(str) == str(r.country_key)]
        gs, bs_ = rr.arm_scores(part, "G"), rr.arm_scores(part, "B2")
        good &= r.keys == len(part)
        good &= all(getattr(r, f"G_{k}") == v for k, v in gs["counts"].items()) and all(getattr(r, f"B2_{k}") == v for k, v in bs_["counts"].items())
        fg = None if gs["binary.f1"] is None else float(gs["binary.f1"])
        fb = None if bs_["binary.f1"] is None else float(bs_["binary.f1"])
        fd = None if gs["binary.f1"] is None or bs_["binary.f1"] is None else float(gs["binary.f1"] - bs_["binary.f1"])
        nan = lambda v: None if pd.isna(v) else v  # noqa: E731
        good &= rr._num_eq(nan(r.f1_G), fg) and rr._num_eq(nan(r.f1_B2), fb) and rr._num_eq(nan(r.f1_G_minus_B2), fd)
    out.append(("country_diagnostics", bool(good)))
    conc = rep["concentration"]
    top3 = float(main["country_key"].value_counts().head(3).sum() / len(main)) if len(main) else None
    shares = (main["target_ord"] // 12).value_counts(normalize=True).sort_index()
    out.append(("concentration", rr._num_eq(conc["top3_country_share"], top3)
                and {str(k): v for k, v in conc["year_shares"].items()} == {str(k): float(v) for k, v in shares.items()}))
    return out, recomputed_status


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, columns: list) -> bool:
    """Exact cell equality (NaN == NaN, numeric/text compared as written) on the given columns."""
    if len(a) != len(b) or any(c not in a.columns for c in columns):
        return False
    for c in columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind in "fiub" and y.dtype.kind in "fiub":
            xf, yf = x.astype(float), y.astype(float)
            if not np.array_equal(xf, yf, equal_nan=True):
                return False
        elif [None if pd.isna(v) else str(v) for v in x] != [None if pd.isna(v) else str(v) for v in y]:
            return False
    return True


def _panel(hi, rows, raw):
    star, phase = projection.project_and_decode(raw)
    return ev.panel(hi.keys["phase_truth"].to_numpy()[rows], phase, hi.Y[rows, 1], star[:, 1], raw[:, 1])


def _indep_f1(truth, phase) -> Fraction:
    """Independent crisis F1 count (pure Python) as an exact fraction."""
    tp = fp = fn = 0
    for t, p in zip(truth.tolist(), phase.tolist()):
        a, b = t >= 3, p >= 3
        tp += a and b
        fp += (not a) and b
        fn += a and not b
    den = 2 * tp + fp + fn
    return Fraction(2 * tp, den) if den else None  # legitimate undefined F1 stays undefined


def _accepted_ok(cand: pd.DataFrame, states: list, contract: dict) -> bool:
    """Re-derive each accepted split from the saved ledger with the strict floor and tie order."""
    rel = float(contract["correction"]["gain_tolerance_relative"])
    for st in states[1:]:
        step = cand[cand["step"] == st["step"]]
        f_cur = float(step["current_objective"].iloc[0])
        tol = rel * max(1.0, abs(f_cur))
        sup = step[step["supported"]]
        best = sup["gain"].max()
        tied = sup[(sup["gain"] >= best - tol) & (sup["gain"] > tol)].sort_values(["node", "descriptor_index", "threshold_order"], kind="mergesort")
        if tied.empty:
            return False
        win = tied.iloc[0]
        accepted = step[step["accepted"]]
        if len(accepted) != 1 or (accepted.iloc[0][["node", "descriptor_index", "threshold_order"]] != win[["node", "descriptor_index", "threshold_order"]]).any():
            return False
    last = states[-1]["step"]
    after = cand[cand["step"] == last + 1]
    if len(after):  # a scored step without acceptance: no supported candidate may be strictly above the floor
        rel_tol = rel * max(1.0, abs(float(after["current_objective"].iloc[0])))
        if ((after["supported"]) & (after["gain"] > rel_tol)).any():
            return False
    return True


def _resolve_state(hi, ppool, pool, transform, pmap, lam, contract):
    leaf_of = pmap.route(hi.D[pool["rows"]], pool["routable"])
    leaves = sorted(pmap.leaves)
    K = len(leaves)
    leaf_G = np.zeros((1, K, cr.N_Q, cr.N_SLOTS, cr.N_SLOTS))
    leaf_b = np.zeros((1, K, cr.N_Q, cr.N_SLOTS))
    leaf_m = np.zeros((1, K))
    for k, leaf in enumerate(leaves):
        m = leaf_of == leaf
        st = cr.Stats.from_rows(ppool.V[m], ppool.R[m], ppool.a[m])
        leaf_G[0, k], leaf_b[0, k], leaf_m[0, k] = st.G, st.b, st.mass
    c = contract["correction"]
    sol = cr.solve_joint_batch(ppool.total, leaf_G, leaf_b, leaf_m, transform.active(), float(c["shared_lambda"]), lam,
                               float(c["solver_residual_tolerance"]))
    return {"objective": float(sol["objective"][0]), "beta": sol["beta"][0]}


def _projection_ok(pred: pd.DataFrame) -> bool:
    from sklearn.isotonic import isotonic_regression

    for arm in ("B0", "B1", "B2", "G"):
        raw = pred[[f"{arm}_{q}_raw" for q in TARGETS]].to_numpy()
        star = pred[[f"{arm}_{q}_star" for q in TARGETS]].to_numpy()
        ref = np.array([np.clip(isotonic_regression(r, increasing=False), 0.0, 1.0) for r in raw])
        if not np.allclose(ref, star, rtol=0, atol=1e-12):
            return False
        if not np.array_equal(1 + (star >= 0.20).sum(axis=1), pred[f"{arm}_phase"].to_numpy()):
            return False
    return True
