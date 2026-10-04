"""Error types. A contract violation stops the run; it is never a fallback."""

from __future__ import annotations


class ContractError(RuntimeError):
    """A pinned identity, structure or frozen-contract check failed."""


class NotImplementedPhaseError(RuntimeError):
    """A scientific phase that has not been ported yet was requested.

    Raised before any output directory is created, so an unported command can
    never leave a success-shaped empty artifact behind.
    """


class TechnicalError(RuntimeError):
    """R41 technical failure: a fit/predict raised, a raw share is NaN/Inf, shapes or
    keys are misaligned, projection failed, a global prefix changed, or a model/map
    artifact is corrupt or conflicting. The affected run stops and is incomplete;
    it is never converted into a normal parent/global fallback."""
