# Smallest production C path and resource envelope (proposal only; nothing started)

## Plan (local enumeration, `c-pilot2-20261010/production_plan_cap16000000.csv.gz`)

`plan_c_requests` over all 6,227 frozen admins with windows ≤ 1.6e7 cells: 57,801 request pieces; 4.6711e11 cells;
per piece median 7.8e6, max 1.6e7 cells; candidates median 2, max 16; 0 zero-source admins; 0 no-source area. Every
piece binds admin, window (lattice cols/rows), ordered candidate ids and the D30 recipe id; the denominator stays the
whole frozen admin area.

## Envelope (from measured pilot rates; EECU cost and quotas unknown)

Per piece ≈ 1 s + cells × (2.0–3.2e-6 s): largest piece ~33–52 s (< 120 s deadline); per-piece cap 129 × cells × 1.1
band-px (≤ 2.27e9). Total ≈ 1.2–1.9e6 s: ~14–22 days sequential, ~1.8–2.8 days at 8 concurrent, ~0.9–1.4 days at 16
— only if the project's interactive concurrency/quota allows (not visible without billing access). Output: ~57,801
small JSON rows (≈ 4 KB each, ~0.25 GB) kept outside Dropbox; compact tables in evidence.

## Smallest code path (to implement after release; no cloud call)

1. `scripts/ee_extract_c.py`: production runner reusing the pilot's pinned `composite`/`reduction`/projection check;
   frozen plan manifest (pieces + geometry + candidates, sha-bound); per-piece ledger with atomic local writes; resume
   only missing/failed pieces; bounded thread concurrency; stop on auth/permission/quota/rate errors, on repeated
   failures (e.g. > 0.5% of attempted pieces or 3 consecutive), and at a released call/wall cap; never enlarges caps.
2. `extraction`: write the production plan manifest and parse runner outputs into the frozen export columns + recipe;
   `validate_export` + `descriptor_table` (already D30-bound) → descriptor table and coverage ledger for 6,227 admins.
3. `descriptors.attach_descriptors`: build X424 from the descriptor table under a fresh run (new prepared bindings;
   no stale-contract reuse).
4. Focused tests: manifest/ledger/resume (fake ee), output parsing round trip, descriptor assembly on a synthetic plan.

## Staged release proposal

S-1: ~200 representative pieces sequential (measure rate, errors) → S-2: ~2,000 pieces at concurrency 4 → S-3: full
plan at the highest concurrency S-2 shows to be error-free (≤ 8 proposed), each stage with its own call/wall caps and a
checkpoint report. All within EE query/download authorization; no GCS/Drive writes. The legacy native
`extract_alphaearth.py` stays comparison-only.
