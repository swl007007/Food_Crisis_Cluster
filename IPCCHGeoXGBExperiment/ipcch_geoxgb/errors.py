"""Error types. A contract violation stops the run; it is never a fallback."""

from __future__ import annotations


class ContractError(RuntimeError):
    """A pinned identity, structure or frozen-contract check failed."""


class NotImplementedPhaseError(RuntimeError):
    """A scientific phase that has not been ported yet was requested.

    Raised before any output directory is created, so an unported command can
    never leave a success-shaped empty artifact behind.
    """
