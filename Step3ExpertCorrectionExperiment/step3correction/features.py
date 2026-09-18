"""Main-pipeline feature assembly with imputation deferred to the fit window.

The correction experiment must keep the *original* main feature definitions and
missing-value conventions, so it calls the production
``src.feature.feature.prepare_features`` unchanged.  Production applies
``comp_impute(strategy="max_plus", multiplier=100.0)`` fitted on the **whole**
panel; a correction fit may not inherit that global statistic.  We therefore
capture the matrix immediately *before* imputation - reproducing production's
``inf -> NaN`` step exactly - and refit the same
:class:`src.customize.customize.OutOfRangeImputer` strategy separately inside
each correction fit/refit window.

``prepare_features`` also writes ``correspondence_table.csv`` and
``feature_columns_debug.csv`` into the process working directory, so callers must
run it from inside the experiment working directory.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .protected import EXPERIMENT_DIR, PANEL_SOURCE, ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _production_imputer_class():
    """Import the production out-of-range imputer lazily."""
    from src.customize.customize import OutOfRangeImputer

    return OutOfRangeImputer


def _production_modules():
    """Import the production panel/feature builders lazily.

    ``src.feature.feature`` pulls in ``polars``, which is only present in the
    project environments.  Deferring the import keeps the pure correction logic
    (windows, selection, abstention, fit-only imputation) testable in any
    environment.
    """
    from src.feature import feature as feature_module
    from src.preprocess.preprocess import load_and_preprocess_data

    return feature_module, load_and_preprocess_data


IMPUTE_STRATEGY = "max_plus"
IMPUTE_MULTIPLIER = 100.0


class FeatureContractError(RuntimeError):
    """Raised when the assembled feature panel violates an alignment contract."""


@dataclass(frozen=True)
class FeaturePanel:
    """Pre-imputation main feature matrix aligned to ``(admin_code, month_start)``."""

    X: np.ndarray
    feature_names: List[str]
    y: np.ndarray
    dates: pd.Series
    admin_codes: np.ndarray
    forecasting_scope: int
    active_lag_months: int

    def provenance(self) -> Dict[str, object]:
        """Return feature provenance for the run manifest."""
        return {
            "panel_source": str(PANEL_SOURCE),
            "n_rows": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "forecasting_scope": int(self.forecasting_scope),
            "active_lag_months": int(self.active_lag_months),
            "imputation": {
                "strategy": IMPUTE_STRATEGY,
                "multiplier": IMPUTE_MULTIPLIER,
                "fitted_on": "correction fit/refit window rows only",
                "production_difference": (
                    "production prepare_features fits comp_impute on the whole panel; "
                    "this experiment defers and refits it inside each window"
                ),
            },
            "feature_order_first": self.feature_names[:5],
            "feature_order_last": self.feature_names[-5:],
        }


@contextmanager
def working_directory(path: Path):
    """Temporarily chdir so production feature side-effect CSVs stay contained."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


def _capture_pre_imputation(store: Dict[str, np.ndarray]):
    """Return a ``comp_impute`` stand-in that records the pre-imputation matrix.

    Reproduces production's ``inf -> NaN`` column sweep (``comp_impute`` lines in
    ``src/preprocess/preprocess.py``) and then returns the still-missing matrix so
    the downstream ``inf -> 0`` guard is a no-op and imputation can be refitted
    per window.
    """

    def capture(X, strategy=IMPUTE_STRATEGY, multiplier=IMPUTE_MULTIPLIER):
        frame = pd.DataFrame(X)
        numeric = frame.apply(pd.to_numeric, errors="coerce").astype(float)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        matrix = numeric.to_numpy(dtype=float)
        store["X_pre_impute"] = matrix
        return matrix

    return capture


def build_feature_panel(
    *,
    forecasting_scope: int,
    panel_path: Path | str = PANEL_SOURCE,
    working_dir: Path | str = EXPERIMENT_DIR / "outputs" / "feature_workdir",
) -> FeaturePanel:
    """Assemble the main pre-imputation feature panel for one forecasting scope."""
    from src.utils.lag_schedules import forecasting_scope_to_lag
    from config import LAGS_MONTHS

    feature_module, load_and_preprocess_data = _production_modules()
    active_lag = forecasting_scope_to_lag(forecasting_scope, LAGS_MONTHS)
    df = load_and_preprocess_data(str(panel_path))

    if "latitude" in df.columns and "longitude" in df.columns:
        X_loc = df[["latitude", "longitude"]].to_numpy()
    elif "lat" in df.columns and "lon" in df.columns:
        X_loc = df[["lat", "lon"]].to_numpy()
    else:
        raise FeatureContractError("Panel must expose latitude/longitude or lat/lon")

    # prepare_features sorts internally; assert the input is already in that order
    # so downstream admin-code alignment cannot drift silently.
    order = df.sort_values(by=["FEWSNET_admin_code", "date"]).index.to_numpy()
    if not np.array_equal(order, df.index.to_numpy()):
        raise FeatureContractError(
            "Panel is not pre-sorted by (FEWSNET_admin_code, date); "
            "row alignment against archived predictions cannot be guaranteed"
        )

    store: Dict[str, np.ndarray] = {}
    original = feature_module.comp_impute
    feature_module.comp_impute = _capture_pre_imputation(store)
    try:
        with working_directory(working_dir):
            X, y, _, _, _, _, dates, feature_columns = feature_module.prepare_features(
                df,
                np.zeros(len(df), dtype=int),
                X_loc,
                forecasting_scope=forecasting_scope,
            )
    finally:
        feature_module.comp_impute = original

    if "X_pre_impute" not in store:
        raise FeatureContractError("prepare_features did not reach the imputation step")
    X_pre = store["X_pre_impute"]
    if X_pre.shape != np.asarray(X, dtype=float).shape:
        raise FeatureContractError("Captured pre-imputation matrix shape mismatch")
    if len(feature_columns) != X_pre.shape[1]:
        raise FeatureContractError("Feature name count does not match matrix width")

    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    admin_codes = (
        df.sort_values(by=["FEWSNET_admin_code", "date"])["FEWSNET_admin_code"]
        .to_numpy()
    )
    y = np.asarray(y, dtype=float)
    if np.isnan(y).any():
        raise FeatureContractError("Unlabeled rows must already be filtered by preprocessing")
    return FeaturePanel(
        X=X_pre,
        feature_names=[str(name) for name in feature_columns],
        y=y.astype(int),
        dates=dates,
        admin_codes=admin_codes,
        forecasting_scope=int(forecasting_scope),
        active_lag_months=int(active_lag),
    )


def fit_window_imputer(X_fit: np.ndarray):
    """Fit the production out-of-range imputer on fit-window rows only."""
    imputer = _production_imputer_class()(strategy=IMPUTE_STRATEGY, multiplier=IMPUTE_MULTIPLIER)
    imputer.fit(np.asarray(X_fit, dtype=float))
    return imputer


def transform_with(imputer, X: np.ndarray) -> np.ndarray:
    """Apply a frozen window imputer, then neutralise any residual non-finite value."""
    transformed = np.asarray(imputer.transform(np.asarray(X, dtype=float)), dtype=float)
    non_finite = ~np.isfinite(transformed)
    if non_finite.any():
        transformed = np.where(non_finite, 0.0, transformed)
    return transformed


def impute_fit_and_apply(
    X_fit: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Fit on ``X_fit`` and transform ``X_fit`` plus every other split."""
    imputer = fit_window_imputer(X_fit)
    return tuple(transform_with(imputer, block) for block in (X_fit, *others))
