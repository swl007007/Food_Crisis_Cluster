"""Model adapters for GeoRF pipeline variants.

These adapters expose a uniform interface so that the main pipeline can swap
between the classic GeoRF backend and the XGBoost-based variant without changing
any other logic or configuration.  Each adapter is responsible for providing the
underlying model class, optional constructor kwargs, and any specialization for
two-layer training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from config import (  # Reuse shared training defaults
    MAX_DEPTH,
    MIN_DEPTH,
    N_JOBS,
)


@dataclass
class BaseAdapter:
    """Base adapter with common helpers for GeoRF-style models."""

    key: str
    display_name: str
    baseline_label: str
    model_cls: Any
    init_kwargs: Dict[str, Any] = field(default_factory=dict)

    def create_model(self, *, max_depth_override: Optional[int] = None) -> Any:
        """Instantiate the underlying model with shared defaults."""

        kwargs = {
            "min_model_depth": MIN_DEPTH,
            "max_model_depth": MAX_DEPTH,
            "n_jobs": N_JOBS,
            "max_depth": max_depth_override,
        }
        kwargs.update(self.init_kwargs)
        return self.model_cls(**kwargs)

    def hyperparameter_summary(self) -> Dict[str, Any]:
        """Return adapter-specific hyperparameters for logging."""

        return {k: v for k, v in self.init_kwargs.items() if k not in {"max_depth"}}

    def two_layer_fit_kwargs(self, *, feature_names_L1: Optional[list] = None) -> Dict[str, Any]:
        """Return kwargs needed when calling ``fit_2layer``."""

        return {}


class GFAdapter(BaseAdapter):
    """Adapter for the classic GeoRF backend."""

    def __init__(self) -> None:
        from src.model.GeoRF import GeoRF  # Local import avoids circular deps

        super().__init__(
            key="gf",
            display_name="GeoRF",
            baseline_label="RF",
            model_cls=GeoRF,
            init_kwargs={},
        )


class XGBAdapter(BaseAdapter):
    """Adapter for the GeoRF pipeline backed by XGBoost."""

    def __init__(self) -> None:
        from src.model.GeoRF_XGB import GeoRF_XGB  # Local import avoids heavy deps at import time

        super().__init__(
            key="xgb",
            display_name="GeoXGB",
            baseline_label="XGB",
            model_cls=GeoRF_XGB,
            init_kwargs={
                "learning_rate": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        )

    def two_layer_fit_kwargs(self, *, feature_names_L1: Optional[list] = None) -> Dict[str, Any]:
        return {"feature_names_L1": feature_names_L1}


class DTAdapter(BaseAdapter):
    """Adapter for the GeoRF pipeline backed by Decision Trees."""

    def __init__(self) -> None:
        from src.model.GeoRF_DT import GeoRF_DT  # Local import avoids circular deps

        super().__init__(
            key="dt",
            display_name="GeoDT",
            baseline_label="DT",
            model_cls=GeoRF_DT,
            init_kwargs={},
        )
