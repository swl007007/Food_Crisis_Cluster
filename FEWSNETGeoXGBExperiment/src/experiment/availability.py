"""Release-aware IPC visibility and original-key A/B fitting views (interruption design D1/D2/G2/G4).

Release ledger contract (one row per published FEWS NET product cycle per country):
``cycle_id, product, country, reference_month, release_date, evidence, source``.
``reference_month`` is the assessment month (``YYYY-MM``), ``release_date`` the eligible public
release (``YYYY-MM-DD``), ``evidence`` one of ``verified_vintage | reconstructed | synthetic``.
``country`` must equal the observations' ``country`` code. A release is visible at the cutoff
of origin X (end of month X) iff its release month <= X. A real run refuses synthetic rows and
any labelled (country, month) without a ledger row: a missing calendar blocks dependent fitting
(D7); nothing here invents a release date.

Cycles are publication identities: ``cycle_id`` maps one-to-one to a reference month, and the
cycles' publication order must agree with their reference order (otherwise the ledger is
refused rather than choosing an order). ``DUE_RULE`` (service-wide reconstruction metadata, not
a sourced D7 fact): a cycle is due at X when at least one country's release of it is visible at
X. Scenario k hides the latest k due cycles for every country (G1). Fewer than k due cycles is
an unsupported case, never a smaller k.

For a forecast at X with intensity k and inherited exclusions E (outer cycles for internal gate
origins), E' = E | hidden(X, k). Actual cases may carry a per-country intensity ``{country: k}``
(verified missed ordinarily-due cycles) and per-country exclusions ``{country: cycles}``: the masks
apply to that country's rows only, inside the same pooled/local fitting and prediction population
(no per-country model). Fitting labels are months in [X-59, X) released by X and not in
E'. A training key's IPC history is rebuilt at its own origin o from releases visible at o,
minus its own hidden(o, k') and minus E': strategy A uses k' = 0 at weight 1, strategy B the
k' = 0/1/2 variants at weight 1/3 each, grouped by original key; support counts original keys.
Prediction keys are given explicitly and never need a label. Evaluator truth is a separate table,
EMPTY unless explicitly supplied (development evaluation, or the separate post-freeze release
of final truth), and never feeds features, fits or cache identities; historical gate truth uses
only input observations lawful at the outer cutoff.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.experiment import plan
from src.feature import fourclass_features as ff
from src.model import native_xgb as nx

LEDGER_COLUMNS = ("cycle_id", "product", "country", "reference_month", "release_date", "evidence", "source")
EVIDENCE = ("verified_vintage", "reconstructed", "synthetic")
TRUTH_PRODUCT = "CS"
STRATEGIES = {"A": ((0, 1.0),), "B": ((0, 1 / 3), (1, 1 / 3), (2, 1 / 3))}
DUE_RULE = "earliest_country_release"


class UnsupportedScenario(ValueError):
    """The documented calendar cannot define the requested missed cycles."""


class ReleaseLedger:
    def __init__(self, frame: pd.DataFrame, real: bool = False):
        missing = [c for c in LEDGER_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"release ledger lacks columns {missing}")
        led = frame.loc[frame["product"] == TRUTH_PRODUCT, list(LEDGER_COLUMNS)].copy()
        if led[list(LEDGER_COLUMNS)].isna().any().any() or (led["source"].astype(str).str.strip() == "").any():
            raise ValueError("release ledger has empty fields")
        if not led["evidence"].isin(EVIDENCE).all():
            raise ValueError(f"ledger evidence must be one of {EVIDENCE}")
        if real and (led["evidence"] == "synthetic").any():
            raise ValueError("synthetic release rows in a real run")
        led["ref"] = ff.month_index(pd.to_datetime(led["reference_month"].map(str), format="%Y-%m"))
        stamp = pd.to_datetime(led["release_date"].map(str), format="%Y-%m-%d")
        led["release"] = (stamp.dt.year * 12 + stamp.dt.month - 1).to_numpy(dtype=np.int64)
        if led.duplicated(["country", "ref"]).any():
            raise ValueError("duplicate (country, reference_month) release rows")
        if (led["release"] < led["ref"]).any():
            raise ValueError("a release precedes its reference month")
        if (led.groupby("ref")["cycle_id"].nunique() != 1).any() or (led.groupby("cycle_id")["ref"].nunique() != 1).any():
            raise ValueError("cycle_id must map one-to-one to a reference month")
        self.real = real
        self.due_rule = DUE_RULE
        self.frame = led.reset_index(drop=True)
        self._release = {(c, int(r)): int(m) for c, r, m in zip(led["country"], led["ref"], led["release"])}
        first = led.groupby("ref")["release"].min()
        if (np.diff(first.to_numpy()) < 0).any():
            raise ValueError("cycle publication order disagrees with reference order")
        self._cycles = first.index.to_numpy(dtype=np.int64)
        self._due_at = first.to_numpy(dtype=np.int64)
        self.cycle_id = dict(zip(led["ref"].astype(int), led["cycle_id"]))

    def release_month(self, countries, months) -> np.ndarray:
        """Release month per (country, reference month); NaN where the ledger has no row."""
        return np.array([self._release.get((c, int(m)), np.nan) for c, m in zip(countries, months)], dtype=float)

    def require_coverage(self, observations: pd.DataFrame) -> None:
        rel = self.release_month(observations["country"], observations["month"])
        if np.isnan(rel).any():
            gaps = observations.loc[np.isnan(rel), ["country", "month"]].drop_duplicates()
            raise ValueError(f"{len(gaps)} labelled (country, month) without a release row, e.g. "
                             f"{gaps.head(3).to_dict('records')}")

    def fully_released(self, month: int, cutoff: int) -> bool:
        """True when every country's release of cycle ``month`` is visible at ``cutoff``."""
        rel = self.frame.loc[self.frame["ref"] == int(month), "release"]
        return bool(len(rel)) and int(rel.max()) <= int(cutoff)

    def hidden(self, origin: int, k: int) -> frozenset:
        """The latest k cycles due at the end of ``origin`` (synchronised across countries)."""
        if k == 0:
            return frozenset()
        due = self._cycles[self._due_at <= origin]
        if len(due) < k:
            raise UnsupportedScenario(f"{len(due)} cycles due at {ff.month_label([origin])[0]}, k={k}")
        return frozenset(int(c) for c in due[-k:])


class Availability:
    """Masked observations, label pools and feature views for one horizon."""

    def __init__(self, observations: pd.DataFrame, ledger: ReleaseLedger, scaffold, schema: dict,
                 horizon: int, alignment: dict | None = None, truth: pd.DataFrame | None = None):
        obs = observations[["area", "month", "country", "class_code"]].copy()
        if obs.duplicated(["area", "month"]).any():
            raise ValueError("duplicate observed area-months")
        if ledger.real:
            if alignment is None:
                raise ValueError("a real availability run needs an explicit covariate alignment")
            ledger.require_coverage(obs)
        tr = (obs.iloc[:0] if truth is None else truth)[["area", "month", "country", "class_code"]].copy()
        if tr.duplicated(["area", "month"]).any():
            raise ValueError("duplicate evaluator truth keys")
        tr["release"] = ledger.release_month(tr["country"], tr["month"])
        self.truth = tr.reset_index(drop=True)
        obs["release"] = ledger.release_month(obs["country"], obs["month"])
        #: labelled rows without a release row are never visible (a real run refuses them above)
        self.unreleased = int(obs["release"].isna().sum())
        self.obs = obs[obs["release"].notna()].reset_index(drop=True)
        self.ledger = ledger
        self.scaffold = scaffold
        self.schema = schema
        self.horizon = int(horizon)
        self.alignment = None if alignment is None else ff.check_alignment(schema, alignment, real=ledger.real)
        self.features = (ff.aligned_feature_names(schema, alignment) if alignment is not None
                         else list(schema["ordered_features"]))
        self.area_country = observations.drop_duplicates("area").set_index("area")["country"].to_dict()
        self.countries = sorted({str(c) for c in observations["country"]} | {str(c) for c in self.truth["country"]})
        # Immutable input identity: copies above are never mutated; any input label, release,
        # ledger or alignment change gives a new digest (cache identities include it). Evaluator
        # truth is deliberately excluded: forecasts never depend on it.
        payload = [self.obs.sort_values(["area", "month"]).to_csv(index=False),
                   ledger.frame[list(LEDGER_COLUMNS)].to_csv(index=False),
                   json.dumps(self.alignment, sort_keys=True, default=str), str(self.horizon),
                   json.dumps(self.features)]
        self.inputs_sha256 = hashlib.sha256("\x1e".join(payload).encode()).hexdigest()

    # -- visibility -------------------------------------------------------------------
    def hidden_for(self, origin: int, k):
        """hidden(origin, k) for an int k, or {country: hidden(origin, k_c)} for a per-country k."""
        if isinstance(k, dict):
            missing = sorted(set(self.obs["country"].astype(str)) - set(map(str, k)))
            if missing:
                raise ValueError(f"per-country intensity lacks countries {missing[:5]}")
            return {str(c): self.ledger.hidden(origin, int(v)) for c, v in k.items()}
        return self.ledger.hidden(origin, int(k))

    def union(self, a, b):
        """Union of two exclusion sets, each global (frozenset) or per-country (dict). A global
        set stays global: in a mixed union it applies to EVERY known country."""
        if not isinstance(a, dict) and not isinstance(b, dict):
            return frozenset(a) | frozenset(b)
        countries = set(self.countries) | {str(c) for d in (a, b) if isinstance(d, dict) for c in d}
        part = lambda d, c: frozenset(d.get(c, frozenset())) if isinstance(d, dict) else frozenset(d)   # noqa: E731
        return {c: part(a, c) | part(b, c) for c in countries}

    def _masked(self, frame: pd.DataFrame, excluded) -> np.ndarray:
        if isinstance(excluded, dict):
            out = np.zeros(len(frame), dtype=bool)
            country = frame["country"].astype(str).to_numpy()
            for c, months in excluded.items():
                out |= (country == str(c)) & frame["month"].isin(list(months)).to_numpy()
            return out
        return frame["month"].isin(list(excluded)).to_numpy()

    def visible(self, cutoff: int, excluded=frozenset()) -> pd.DataFrame:
        o = self.obs
        return o[(o["release"].to_numpy() <= cutoff) & ~self._masked(o, excluded)]

    def label_pool(self, origin: int, excluded=frozenset()) -> pd.DataFrame:
        """Lawful fitting labels for a fit at ``origin``: months in [O-59, O), released, not excluded."""
        v = self.visible(origin, excluded)
        return v[(v["month"] >= origin - plan.WINDOW) & (v["month"] < origin)]

    def gate_dates(self, origin: int, k: int) -> list:
        """Up to six latest globally observed label months U < O lawful at O after its own mask (G2);
        not restricted to the fitting window."""
        months = self.visible(origin, self.hidden_for(origin, k))["month"].unique()
        earlier = sorted(int(m) for m in months if m < origin)
        return earlier[-plan.GATE_DATES:]

    def provenance(self, origins) -> pd.DataFrame:
        """Source period, evidenced release month (NaN without evidence) and age per aligned
        monthly/annual source at each origin: metadata, never predictors."""
        origins = np.asarray(origins, dtype=np.int64)
        out = {}
        for name, rule in (self.alignment or {}).items():
            if rule["kind"] == "annual":
                ref = ff.annual_reference_year(origins, rule)
                out[f"prov_{name}_ref_year"] = ref
                out[f"prov_{name}_release_month"] = ff.source_release_month(rule, ref)
                out[f"prov_{name}_age_years"] = origins // 12 - ref
            elif rule["kind"] == "monthly":
                src = (origins - rule["lag"]).astype(float)
                out[f"prov_{name}_source_month"] = src
                out[f"prov_{name}_release_month"] = ff.source_release_month(rule, src)
                out[f"prov_{name}_age_months"] = origins - src
        return pd.DataFrame(out, index=range(len(origins)))

    # -- features ---------------------------------------------------------------------
    def key_features(self, areas, targets, inner_k, excluded=frozenset()) -> pd.DataFrame:
        """Features of keys (area, target) at origin T - H; IPC history rebuilt per (origin, k')."""
        areas = np.asarray(areas, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        if isinstance(inner_k, dict):   # per-country intensity -> per-row k of each area's country
            # areas without any labelled history have no country and no IPC inputs to mask
            inner_k = np.array([int(inner_k.get(str(self.area_country.get(int(a))), 0)) for a in areas],
                               dtype=np.int64)
        inner_k = np.broadcast_to(np.asarray(inner_k, dtype=np.int64), areas.shape)
        origins = targets - self.horizon
        if len(areas) == 0:   # an empty (e.g. support-limited) pool is a valid, featureless view
            return pd.DataFrame({c: pd.Series(dtype=float) for c in self.features})
        history = None
        for o, k in sorted(set(zip(origins.tolist(), inner_k.tolist()))):
            pos = np.where((origins == o) & (inner_k == k))[0]
            seen = self.visible(o, self.union(excluded, self.ledger.hidden(o, k)))
            block = ff.history_features(seen.assign(phase=seen["class_code"] + 1)[["area", "month", "phase"]],
                                        areas[pos], origins[pos])
            if history is None:
                history = pd.DataFrame(np.nan, index=range(len(areas)), columns=block.columns)
            history.iloc[pos] = block.to_numpy()
        covariates = ff.covariate_features(self.scaffold, self.schema, areas, targets, origins, self.alignment)
        return pd.concat([covariates, history], axis=1)[self.features]

    def fitting_view(self, origin: int, k: int, strategy: str, excluded=frozenset()):
        """Grouped fitting rows for a fit at ``origin`` under intensity k; returns (frame, support).

        ``support`` counts original keys only (rows/areas/dates/classes), never variant copies."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {sorted(STRATEGIES)}")
        outer = self.union(excluded, self.hidden_for(origin, k))
        pool = self.label_pool(origin, outer).sort_values(["area", "month"]).reset_index(drop=True)
        keys = pd.DataFrame({"orig_key": np.arange(len(pool)), "area": pool["area"].to_numpy(),
                             "country": pool["country"].to_numpy(), "target_month": pool["month"].to_numpy(),
                             "origin_month": pool["month"].to_numpy() - self.horizon,
                             "class_code": pool["class_code"].to_numpy(dtype=np.int64)})
        parts = []
        for variant, weight in STRATEGIES[strategy]:
            parts.append(keys.assign(variant_k=variant, weight=weight))
        rows = pd.concat(parts, ignore_index=True).sort_values(["orig_key", "variant_k"], kind="stable")
        rows = rows.reset_index(drop=True)
        feats = self.key_features(rows["area"], rows["target_month"], rows["variant_k"], outer)
        frame = pd.concat([rows.assign(horizon=self.horizon), self.provenance(rows["origin_month"]), feats], axis=1)
        support = nx.support(keys["class_code"], keys["area"], keys["target_month"])
        return frame, support

    def prediction_view(self, areas, targets, k: int, excluded=frozenset()) -> pd.DataFrame:
        """Feature rows for explicit prediction keys at their origin T - H under intensity k.

        No label is required. Adds the lawful persistence comparator: the latest visible label
        with its source month and age (prolonged-lag persistence when the exact origin is hidden)."""
        areas = np.asarray(areas, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        origins = targets - self.horizon
        if len(set(origins.tolist())) > 1:
            raise ValueError("one prediction view per origin")
        if len(areas) == 0:
            cols = ["area", "country", "target_month", "origin_month", "horizon", "scenario_k",
                    "persistence_class_code", "persistence_source_month", "persistence_age"]
            return pd.concat([pd.DataFrame({c: pd.Series(dtype=float) for c in cols}), self.provenance([]),
                              self.key_features([], [], k)], axis=1)
        feats = self.key_features(areas, targets, k, excluded)
        row_k = ([int(k.get(str(self.area_country.get(int(a))), 0)) for a in areas] if isinstance(k, dict)
                 else int(k))   # the intensity actually applied to each row
        keys = pd.DataFrame({"area": areas, "country": [self.area_country.get(int(a)) for a in areas],
                             "target_month": targets, "origin_month": origins, "horizon": self.horizon,
                             "scenario_k": row_k})
        seen = self.visible(int(origins[0]), self.union(excluded, self.hidden_for(int(origins[0]), k)))
        latest = seen.sort_values("month").groupby("area").tail(1).set_index("area")
        src = latest["month"].reindex(areas).to_numpy(dtype=float)
        keys["persistence_class_code"] = latest["class_code"].reindex(areas).to_numpy(dtype=float)
        keys["persistence_source_month"] = src
        keys["persistence_age"] = origins - src
        return pd.concat([keys, self.provenance(origins), feats], axis=1)

    def stage1_input(self, target: int, k: int, strategy: str) -> pd.DataFrame:
        """Prepared rows of one scenario Stage 1 root at T (origin O = T - H), design G4.

        role ``fit_variant``: the lawful grouped fitting rows (strategy variants and weights);
        role ``eval``: one row per original key with its designated-k features at its own origin
        and the outer exclusion (S/C evaluation, never augmented); role ``target``: the E3 rows
        at T under k for areas with a genuine target label. ``orig_key`` links fit/eval rows."""
        origin = int(target) - self.horizon
        outer = self.ledger.hidden(origin, k)
        fit, _ = self.fitting_view(origin, k, strategy)
        keys = fit.drop_duplicates("orig_key").sort_values("orig_key").reset_index(drop=True)
        cols = ["orig_key", "area", "country", "target_month", "origin_month", "class_code"]
        ev = pd.concat([keys[cols].assign(variant_k=k, weight=1.0, horizon=self.horizon),
                        self.provenance(keys["origin_month"]),
                        self.key_features(keys["area"], keys["target_month"], k, outer)], axis=1)
        if self.truth.empty:
            raise ValueError("Stage 1 E3 needs development evaluator truth (pass truth=...)")
        labelled = self.truth[self.truth["month"] == target]   # may be empty: recorded downstream
        view = self.prediction_view(labelled["area"].to_numpy(), np.full(len(labelled), target), k)
        tgt = view.assign(orig_key=-1, class_code=labelled["class_code"].to_numpy(dtype=np.int64),
                          variant_k=k, weight=1.0)
        frames = [fit.assign(role="fit_variant"), ev.assign(role="eval"), tgt.assign(role="target")]
        head = ["role"] + cols + ["variant_k", "weight", "horizon"]
        meta = [c for c in fit.columns if c.startswith("prov_")]
        out = pd.concat([f[head + meta + self.features] for f in frames], ignore_index=True)
        return out.assign(scenario_k=k, strategy=strategy)

    def lawful_labels(self, month: int, cutoff: int, excluded=frozenset()) -> pd.DataFrame:
        """Input observations at ``month`` released by ``cutoff`` and not excluded: historical gate
        truth (evaluator-only at the internal origin, lawful at the outer cutoff)."""
        v = self.visible(cutoff, excluded)
        return v[v["month"] == month].sort_values("area").reset_index(drop=True)

    def truth_view(self, areas, targets, lawful_at: int | None = None, excluded=frozenset()) -> pd.DataFrame:
        """Evaluator-only truth for keys: class code or NaN with a reason (never a feature input).

        ``lawful_at`` restricts truth to labels released by that cutoff and not in ``excluded``
        (historical gate truths at the outer cutoff); None = any genuine label."""
        idx = pd.MultiIndex.from_arrays([np.asarray(areas, dtype=np.int64), np.asarray(targets, dtype=np.int64)])
        obs = self.truth.set_index(["area", "month"])
        code = obs["class_code"].reindex(idx).to_numpy(dtype=float)
        release = obs["release"].reindex(idx).to_numpy(dtype=float)
        reason = np.where(np.isnan(code), "no_label", "")
        if lawful_at is not None:
            keyed = pd.DataFrame({"month": idx.get_level_values(1),
                                  "country": [self.area_country.get(int(a), "") for a in idx.get_level_values(0)]})
            hidden = self._masked(keyed, excluded)
            late = ~np.isnan(code) & (np.isnan(release) | (release > lawful_at) | hidden)
            reason = np.where(late, "not_lawful_at_cutoff", reason)
            code = np.where(late, np.nan, code)
        return pd.DataFrame({"area": idx.get_level_values(0), "target_month": idx.get_level_values(1),
                             "truth_code": code, "truth_reason": reason})
