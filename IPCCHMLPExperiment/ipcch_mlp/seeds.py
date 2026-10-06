"""Identity-derived seeds (design section 6): SHA256 of canonical JSON, never Python hash()."""

from __future__ import annotations

import hashlib
import json

STREAMS = ("init", "train", "perm")


def canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def digest(payload: dict) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def derive(seed_identity: dict, stream: str) -> int:
    """63-bit seed for one labelled stream of one scalar model."""
    if stream not in STREAMS:
        raise ValueError(f"unknown seed stream {stream!r}")
    h = hashlib.sha256((canonical(seed_identity) + "|" + stream).encode("utf-8")).hexdigest()
    return int(h[:16], 16) & ((1 << 63) - 1)


def seed_triplet(seed_identity: dict) -> dict:
    return {s: derive(seed_identity, s) for s in STREAMS}
