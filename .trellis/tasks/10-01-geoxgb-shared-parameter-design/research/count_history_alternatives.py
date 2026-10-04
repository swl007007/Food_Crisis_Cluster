"""Count historical label support; no feature construction or model fitting.

Run from repository root: python3 <this-file> > /tmp/history-alternatives.json
These hypothetical windows are not an approved validation schedule.
"""
import csv
import gzip
import json
import statistics
from pathlib import Path

RUN = Path("FEWSNETFourClassBaseline/runs/fourclass-v7-20260928")


def month_index(value: str) -> int:
    year, month = map(int, value.split("-"))
    return year * 12 + month - 1


def describe(values: list[int]) -> dict:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def main() -> None:
    with (RUN / "prepared/ledgers/observations.csv").open() as handle:
        observations = list(csv.DictReader(handle))
    rows = [(int(r["area"]), int(r["month"]), int(r["class_code"])) for r in observations]
    assert len({(a, m) for a, m, _ in rows}) == len(rows)
    with (RUN / "stage2/experiment/knn_sparsification_results/cluster_mapping_k40_nc13_general.csv").open() as handle:
        mapping = {int(r["FEWSNET_admin_code"]): int(r["cluster_id"]) for r in csv.DictReader(handle)}
    groups = []
    for stage in (1, 3):
        for horizon in (4, 8, 12):
            folds = []
            pattern = "stage1/folds/*/candidate.json" if stage == 1 else f"stage3/h{horizon}/folds/*/fold.json"
            for path in sorted(RUN.glob(pattern)):
                fold = json.loads(path.read_text())
                if fold["horizon"] != horizon or (stage == 3 and fold["status"] != "fitted"):
                    continue
                areas = None
                if stage == 1:
                    with gzip.open(path.parent / "fold_membership.csv.gz", "rt") as handle:
                        areas = {int(r["area"]) for r in csv.DictReader(handle) if r["role"] == "heldout_target"}
                folds.append((month_index(fold["origin_month"]), areas))
            for mode, width, shift in (
                ("full35", 35, 0), ("full59", 59, 0), ("full71", 71, 0),
                ("earliest_E1_rolling35", 35, 12 + horizon),
            ):
                counts, months, local_counts = [], [], []
                class4_zero = 0
                for origin, areas in folds:
                    upper = origin - shift
                    selected = [(a, m, c) for a, m, c in rows if upper - width <= m < upper and (areas is None or a in areas)]
                    counts.append(len(selected))
                    months.append(len({m for _, m, _ in selected}))
                    if stage == 3:
                        local, class4 = [0] * 13, [0] * 13
                        for area, _, label in selected:
                            if area in mapping:
                                local[mapping[area]] += 1
                                class4[mapping[area]] += label == 3
                        class4_zero += sum(n == 0 for n in class4)
                        local_counts.extend(local)
                entry = {"stage": stage, "horizon": horizon, "mode": mode, "folds": len(folds), "pooled_rows": describe(counts), "observed_months": describe(months)}
                if stage == 3:
                    entry.update(cluster_fold_denominator=13 * len(folds), class4_zero_cluster_folds=class4_zero, local_rows=describe(local_counts))
                groups.append(entry)
    output = {"scope": "count-only hypothetical historical support; no feature availability or label release dates checked", "ledger_first": min(r["month_label"] for r in observations), "ledger_last": max(r["month_label"] for r in observations), "groups": groups}
    evidence = Path(__file__).with_name("sample-support-counts.json")
    if evidence.exists():
        assert output == json.loads(evidence.read_text())["history_alternatives"]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
