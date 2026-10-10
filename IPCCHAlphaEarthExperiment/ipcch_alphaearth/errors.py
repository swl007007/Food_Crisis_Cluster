"""Error types. A contract or technical failure stops the stage; it is never a scientific fallback."""

from __future__ import annotations


class ContractError(RuntimeError):
    """A pinned identity, frozen constant, key/cohort or temporal-ancestry check failed."""


class TechnicalError(RuntimeError):
    """Numerical/IO failure: nonfinite prediction, failed solve, corrupt or conflicting artifact,
    incomplete extraction. Never converted into B2 reuse, low coverage or missing data."""
