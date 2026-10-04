# Evaluator-only October-2025 truth release: CANDIDATE (pending coordinator approval)

2026-10-03, Claude executor, after the coordinator accepted the actual freeze (`actual.json` c896a583…a84a). The 2025 CS values were read for evaluation only. They are never fed to fitting or selection. No scores have been computed and nothing was fitted.

## Packet

Directory `C:\Users\swl00\geoxgb_runs\scen-b43ef6a-v1.truth-release-v1\` (new, written once):

| File | sha256 |
|---|---|
| `crosswalk_oct2025.csv`: all 5,573 raw Oct-2025 CS rows, each with its status and provenance | ab629b01f72ae5696dbbd36a2cc4b1b2707c1233f4cfbdc73d9b3ca04bd3652a |
| `truth_oct2025.csv`: admitted keys (area, target_month 2025-10, class_code 0..3, raw_phase, is_allowing_for_assistance, fnid, source_id) | 7ccc336f55c6325fcf3ccfa7508e2f4a4e2c97a6ecbd5ab493eaf37ab73547f2 |
| `release.json`: `approved: false` (scen-evaluate refuses until approved); truth sha; `frozen_actual` = c896a583…a84a; source shas; rules; June/expert routes | e80f960d02b05c2d4ea1c2c0cabfb33f96dfc9d56ffa953c227b5afa12da495e |
| `release_summary.json` (copy: `research/truth_release_candidate_summary.json`) | b9cf0cdda83a58e29cb845bb4e790de36b5dee8c3785033c94d89ab689ffe41a |
| Builder `research/probes/build_truth_release.py` | ab96e1a767979eb21f25b2857d3a817e34661172bb10d21c7c843a91ff54522b |

**Sources:**
- the raw `Outcome/FEWSNET_IPC/2025_2026_FEWSNET.csv` (sha a64ed4bb…7fce), scenario CS, reporting_date 2025-10. This is **not** the derived, zero-filled `FEWS_2025.csv`.
- `FEWSNET.csv` (pinned 8fdd4cca…) for historical admin names.
- The pinned shapefile DBF (3aba66a6…) for canonical `admin_name`.

## Admission rule

The crosswalk uses exact full-name equality only. A row is admitted when all of the following hold:
1. Its `geographic_unit_full_name` equals a FEWSNET.csv `admin_name` that maps to **exactly one** admin code (all history).
2. That code receives exactly one October row (one-to-one).
3. The raw `country_code` equals the code's country in the prepared observations.
4. The canonical DBF `admin_name` for the code equals the name.
5. The phase is a genuine IPC 1–5; class_code = min(phase, 4) − 1.

There is no fuzzy or spatial repair, no zero fill and no inferred label.

## Result (no scores)

| Status | Rows |
|---|---|
| **admitted** | **4,457**, all inside the 5,718-area prediction universe (77.9% of October forecast keys) |
| excluded: unmatched name | 1,093. DRC 345 (**DRC wholly unmatched**); **Ethiopia 645** (496 admitted); Sudan 30; Nigeria 21; Mali 20; Somalia 19; Malawi 8; Kenya 3; Chad 2 |
| excluded: ambiguous name → several codes | 1 (Kenya "…Kerio Delta, Turkana Central…" → 2995/2996) |
| excluded: DBF name mismatch | 1 (Kenya "…Kibish, Turkana North…": FEWSNET.csv history maps it to 2996, but the DBF name of 2996 is the Kerio Delta unit). It belongs to the same 2995/2996 ambiguity and is kept excluded |
| excluded: no genuine phase | 21 (19 Not Projected, 2 Not Available: Malawi 20, Zimbabwe 1) |

- **Admitted class counts:** 0: 1,201 · 1: 1,796 · 2: 1,250 · 3: 210. Raw phases are 1–4 only; the 2 phase-5 raw rows are among the unmatched rows.
- **Countries with no raw October-2025 CS row:** Uganda. Its keys are coverage-only, with no truth row.
- **Assistance flag:** 44 admitted rows are flagged `is_allowing_for_assistance = True` (113 in the raw rows). They carry the published CS phase, consistent with the historical label definition: `fews_ipc` is the published phase and `fews_ha` is a separate flag. **Proposal:** include them, keeping the flag column so an exclusion would be a simple filter. This is a decision point for the coordinator.

## Disclosures

- **Name equality is key evidence only, not certified geometric identity.** The 2025 boundaries (fnid) may differ from the historical units; no boundary or geometry comparison was performed.
- **June 2025:** no local genuine June-2025 CS source exists. The June targets (H4 and H8) stay forecast/coverage-only with no truth rows. scen-evaluate reports them as unevaluable (`no released genuine truth for this target`).
- **Expert:** no documented same-horizon expert table exists. Keys keep `no_documented_expert_table`, giving explicit NA routes.
- **Missing exact-origin truth:** Study 2 at 2025 origins needs truth at (area, T − H). For October H4 the origin is 2025-06, and for October H8 it is 2025-02. Neither month has released genuine CS, so October Study 2 keys lacking exact-origin truth are excluded and counted under the spec rule, never filled.

## To approve

Set `approved: true` and `approved_by` in a new release.json. The current file is preserved; the approved version gets a new sha. Then run scen-evaluate with `--truth-release <dir>`.
