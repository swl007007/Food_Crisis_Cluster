"""Run-level source freeze (design sections 6, 8, 9).

A run's preflight records the complete package source inventory (all modules
and configs). predict/report/replay compare it with the current inventory and
stop on any difference. The only exception is an explicit, pre-written
``<run>/source-reconciliation.json`` listing each changed file with its old
and new SHA256, a reason and who authorized it; fit-defining files
(``runtime.FIT_SOURCES``) can never be reconciled this way. The preflight
record itself is never rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

from ipcch_yearly_xgb import runtime
from ipcch_yearly_xgb.errors import ContractError


def check_sources(run_dir: Path, saved: dict, current: dict | None = None) -> dict:
    current = current if current is not None else runtime.source_inventory()
    diff = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
    if not diff:
        return {"status": "identical", "files": len(current)}
    path = Path(run_dir) / "source-reconciliation.json"
    if not path.is_file():
        raise ContractError(f"source inventory differs from the run's preflight: {diff}")
    changes = json.loads(path.read_text(encoding="utf-8")).get("changes", {})
    bad = []
    for k in diff:
        e = changes.get(k)
        if k in runtime.FIT_SOURCES:
            bad.append(f"{k}: fit-defining source cannot be reconciled")
        elif not e or e.get("old") != saved.get(k) or e.get("new") != current.get(k) or not e.get("reason") \
                or not e.get("authorized_by"):
            bad.append(f"{k}: no matching authorized reconciliation entry")
    if bad:
        raise ContractError(f"unreconciled source changes: {bad}")
    return {"status": "reconciled", "changes": diff}
