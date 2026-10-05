"""Zero-fit D29 error-change tally (stdlib only): root vs frozen full tree on saved S/C/E3 hard
predictions, binary crisis = four-class code >= 2. Usage: python3 d29_error_changes.py RUN_STAGE OUT_CSV"""
import csv, gzip, json, os, sys
from collections import defaultdict
stage, out = sys.argv[1], sys.argv[2]

def rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)

def tally(acc, y, r, l):
    y, r, l = int(y) >= 2, int(r) >= 2, int(l) >= 2
    acc["n"] += 1; acc["positive"] += y
    acc["corrected"] += (r != y) and (l == y); acc["spoiled"] += (r == y) and (l != y)
    acc["new_tp"] += y and not r and l; acc["lost_tp"] += y and r and not l
    acc["new_fp"] += (not y) and (not r) and l; acc["removed_fp"] += (not y) and r and (not l)

table = defaultdict(lambda: defaultdict(int))
for cand in sorted(os.listdir(f"{stage}/candidates")):
    cd = f"{stage}/candidates/{cand}"
    root = next(f"{stage}/roots/{r}" for r in os.listdir(f"{stage}/roots")
                if json.load(open(f"{stage}/roots/{r}/completion.json")).get("candidates") == [cand])
    s_rows = defaultdict(int)
    for m in rows(f"{root}/fold_membership.csv.gz"):
        if m["role"] == "validation": s_rows[m["area"]] += 1
    sources = {"S": (f"{cd}/validation_predictions.csv.gz", "area", "y_true", "y_root", "y_final"),
               "C": (f"{cd}/confirmation_predictions.csv.gz", "area", "y_true", "y_root", "y_final"),
               "E3": (f"{cd}/target_predictions.csv", "FEWSNET_admin_code", "y_true_code", "y_pred_pooled_code", "y_pred_partitioned_code")}
    for role, (path, ak, yk, rk, lk) in sources.items():
        for x in rows(path):
            b = x["branch_id"]; depth = 0 if b == "root" else len(b)
            for group in ("all", f"S_rows_{min(s_rows.get(x[ak], 0), 2)}", f"depth_{depth}"):
                tally(table[(cand, role, group)], x[yk], x[rk], x[lk])
keys = ["n", "positive", "corrected", "spoiled", "new_tp", "lost_tp", "new_fp", "removed_fp"]
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["candidate", "role", "group"] + keys)
    for (c, r, g), a in sorted(table.items()): w.writerow([c, r, g] + [a[k] for k in keys])
tot = defaultdict(lambda: defaultdict(int))
for (c, r, g), a in table.items():
    for k in keys: tot[(r, g)][k] += a[k]
for (r, g) in sorted(tot):
    if r == "E3": print(r, g, {k: tot[(r, g)][k] for k in keys})
