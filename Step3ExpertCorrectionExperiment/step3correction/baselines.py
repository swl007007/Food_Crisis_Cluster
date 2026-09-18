"""Reused frozen Stage 3 baselines (pooled for fs1/fs2, both methods for fs3).

Design decision recorded explicitly: pooled and fs3 predictions are **reused**
from the frozen archive rather than retrained.  The installed environment differs
from the historical package environment, so refitting an already frozen
comparator could not be verified bit-for-bit.  Every reused array is checked for
archive/package hash equality, per-row equality and archived-metric reproduction
before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .expert import counts_and_scores
from .protected import archive_stage3_path, package_stage3_path, sha256

KEYS = ["admin_code", "month_start"]
EXPECTED_SUPPORT_ROWS = 62189
EXPECTED_MONTHS = pd.to_datetime(
    [f"{year}-{month:02d}-01" for year in range(2021, 2025) for month in (2, 6, 10)]
)
METRIC_COLUMNS = ("precision", "recall", "f1")


class BaselineContractError(RuntimeError):
    """Raised when a reused frozen baseline fails an equality or metric check."""


@dataclass(frozen=True)
class ReusedBaseline:
    """Frozen Stage 3 predictions plus the checks that validated them."""

    scope: int
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    checks: Dict[str, object]


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise archived prediction keys to the experiment convention."""
    frame = frame.rename(columns={"FEWSNET_admin_code": "admin_code"}).copy()
    frame["admin_code"] = pd.to_numeric(frame["admin_code"], errors="raise").astype("int64")
    frame["month_start"] = pd.to_datetime(frame["month_start"], errors="raise")
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise BaselineContractError("Invalid or duplicate archived prediction key")
    return frame.sort_values(KEYS).reset_index(drop=True)


def load_reused_baseline(scope: int) -> ReusedBaseline:
    """Load, cross-check and metric-verify one frozen Stage 3 scope."""
    checks: Dict[str, object] = {"scope": scope, "source": "frozen archive (not retrained)"}

    hash_pairs = {}
    for name in ("predictions_monthly.csv", "metrics_monthly.csv", "run_manifest.json"):
        package_digest = sha256(package_stage3_path(scope, name))
        archive_digest = sha256(archive_stage3_path(scope, name))
        if package_digest != archive_digest:
            raise BaselineContractError(
                f"fs{scope} {name}: package and archive copies differ "
                f"({package_digest} vs {archive_digest})"
            )
        hash_pairs[name] = package_digest
    checks["archive_package_hashes_match"] = True
    checks["hashes"] = hash_pairs

    predictions = _normalize(pd.read_csv(package_stage3_path(scope, "predictions_monthly.csv")))
    archive_copy = _normalize(pd.read_csv(archive_stage3_path(scope, "predictions_monthly.csv")))
    pd.testing.assert_frame_equal(predictions, archive_copy)
    checks["per_row_archive_equality"] = True

    if len(predictions) != EXPECTED_SUPPORT_ROWS:
        raise BaselineContractError(
            f"fs{scope} support is {len(predictions)} rows, expected {EXPECTED_SUPPORT_ROWS}"
        )
    if set(predictions["month_start"]) != set(EXPECTED_MONTHS):
        raise BaselineContractError(f"fs{scope} evaluation schedule differs from Feb/Jun/Oct 2021-2024")
    checks["support_rows"] = int(len(predictions))
    checks["evaluation_months"] = [str(m.date()) for m in sorted(set(predictions["month_start"]))]

    metrics = pd.read_csv(package_stage3_path(scope, "metrics_monthly.csv"))
    if metrics.duplicated(["test_month", "model"]).any():
        raise BaselineContractError(f"fs{scope} archived metrics have duplicate rows")

    reproduced: List[str] = []
    for model in ("pooled", "partitioned"):
        for month, group in predictions.groupby("month_start"):
            counts, scores = counts_and_scores(group["y_true"], group[f"y_pred_{model}"])
            match = metrics.loc[
                metrics["model"].eq(model) & metrics["test_month"].eq(month.strftime("%Y-%m"))
            ]
            if len(match) != 1:
                raise BaselineContractError(
                    f"fs{scope} archived metrics missing {model} {month:%Y-%m}"
                )
            row = match.iloc[0]
            np.testing.assert_array_equal(counts, row[["tp", "fp", "fn", "tn"]].to_numpy())
            np.testing.assert_allclose(
                scores,
                row[list(METRIC_COLUMNS)].to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
            )
            if int(row["n"]) != len(group):
                raise BaselineContractError(f"fs{scope} archived n mismatch for {model}")
            reproduced.append(f"{model}:{month:%Y-%m}")
    checks["archived_metric_rows_reproduced"] = len(reproduced)
    return ReusedBaseline(scope=scope, predictions=predictions, metrics=metrics, checks=checks)
