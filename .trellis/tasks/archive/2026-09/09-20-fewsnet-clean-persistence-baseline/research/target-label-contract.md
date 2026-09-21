# Target and observed-history contract — read-only evidence

Planning inspection on 2026-09-20. A bounded scout streamed both complete CSVs;
a separate scout traced label/history code. The main session checked the
ledger's terminal physical lines and the legacy source_truth expression.
No source files were changed, no model was fitted and no performance was scored.
The counts/hashes below are scout-reported scan results, not a second independent
full scan by the main session. Validation policy below is approved under D63.

## Source identity and scan method

Paths relative to Analysis/1.Source Data/:

| Source | Path | Bytes | SHA-256 |
|---|---|---:|---|
| M | FEWSNET_forecast_unadjusted_bm.csv | 716303754 | 611f9e776380e28da3fc845888d66a117626f91b37868e236ce849d44bc8f651 |
| L | Outcome/FEWSNET_IPC/FEWSNET.csv | 51422500 | 8fdd4cca6f6ba26b84efc209c8eb36492e1257d51e24edd2c2ad4962df7b38d0 |

M mtime UTC: 2025-11-05T14:41:30.068928+00:00.
L mtime UTC: 2025-03-14T14:14:41+00:00.
Size/mtime were unchanged before/after the main scan; raw-byte SHA-256 was
computed separately. Python3 stdlib csv.reader used utf-8-sig and newline='',
explicit field-width/key checks, raw value counters and all occurrences per
canonical key, with no deduplication. Prior reader.line_num+1 tracked record
start physical lines, including quoted multiline records.

M has 88 columns; keys are FEWSNET_admin_code and date (YYYY-MM). L has 16
columns; keys are admin_code and year_month (YYYY_MM), with redundant year/month.
Compare integer area code and normalized calendar month. All valid area strings
are canonical nonnegative decimal integers, with no normalization collisions.
No valid L date disagrees with its year/month. No area has multiple country
labels within either source. L has no binary target column.

| Count | M | L |
|---|---:|---:|
| Data records, excluding header and including malformed record | 1,029,240 | 302,949 |
| Valid-key records / unique keys | 1,029,240 | 302,948 |
| Duplicate keys / excess duplicate rows | 0 / 0 | 0 / 0 |
| Malformed keys | 0 | 1 |
| Wrong-width records | 0 | 1 |
| Unique areas on valid keys | 5,718 | 5,716 |
| Unique months | 180 | 53 |
| Date extent | 2010-01 to 2024-12 | 2009-07 to 2024-10 |

## Observed target and phase reconciliation

| Raw unadjusted phase | M rows | L rows |
|---|---:|---:|
| 1.0 | 133,211 | 137,113 |
| 2.0 | 82,907 | 85,268 |
| 3.0 | 39,062 | 41,169 |
| 4.0 | 4,186 | 4,601 |
| 5.0 | 74 | 74 |
| Empty field | 769,800 | 34,723 |
| Field absent in malformed record | 0 | 1 |

M fews_ipc_crisis contains exactly 216,118 zeros, 43,322 ones and 769,800
empty fields. All 259,440 observed binary labels have valid integral phases
1..5 and equal 1[fews_ipc>=3]. Phase/binary missingness agrees on every row.
There are no other phase or binary values in valid records. This verifies
current unadjusted contents, not the historical master-generation execution.

Over all 1,029,240 M keys:

| Relation to L | Count |
|---|---:|
| Same key, nonmissing phase agrees | 259,440 |
| Same key, both phases missing | 32,076 |
| Same key, one phase missing | 0 |
| Same key, conflicting nonmissing phases | 0 |
| L key absent, M phase missing | 737,724 |
| L key absent, M phase observed | 0 |

Overlap is 291,516 keys. L has 11,432 additional keys, exactly 5,716 each in
2009-07 and 2009-10. These are earlier observed-history records, not authority
to extend the master target cohort. Their history use remains subject to the
approved per-row origin cutoffs. No duplicate keys make these comparisons
ambiguous. Key presence does not imply a valid observed phase.

Examples (physical CSV lines): area 0, 2010-01 agrees at M:2 and L:244334
(phase 2.0, master binary 0); area 0, 2020-10 has both phases missing at
M:131 and L:244372. Area 0, 2010-02 at M:3 has missing phase/binary and no L key.

## The one non-observation artifact

L:303003, its final physical line, is exactly one field:

    System.IO.MemoryStream

It has no area/date/phase fields and width 1 instead of 16. It is data record
302,949 excluding the header (logical CSV record 302,950 including the header).
There are 53 embedded extra physical lines in earlier CSV records. Thus earlier
notes saying row 302950 are logical-record references, not physical-line locations.
The main session directly verified L:303001-303003. Do not count this artifact
as an ordinary missing-phase area-month observation.

## Existing code and lineage limits

- Step3ExpertCorrectionExperiment/step3correction/expert.py:185-203 parses
  numeric keys, drops rows whose key fields are all missing and then applies
  raw['fews_ipc'].ge(3).astype(int). It does not enforce the phase set 1..5:
  NaN becomes zero, and a hypothetical sentinel 99 would become one.
- PersistenceCorrectionExperiment/persistencecorrection/persistence.py:192-206
  retains phase_raw/phase_missing while copying that source_truth into crisis.
  Its :253-285 join checks source-key coverage and exact horizon, not valid
  observed phase. This explains the missing-phase persistence-zero route
  already rejected by D36; an entirely absent source key instead raises.
- expert.py:122-139 rejects duplicate canonical area/month keys. Existing
  runner truth-agreement checks compare derived binary values, not proof of
  observed-phase validity (step3correction/runner.py:194-200).
- GeoRFBaseline/src/preprocess/preprocess.py:166-197 reads the supplied binary
  label; it does not establish its generator. Adjacent assembly notebooks
  under Analysis/2.source_code/Step4_append_granular_df map adjusted phase to
  binary (02_examine_and_lag_FEWSNET_df.ipynb:90-95 and
  03_drop_IPC_columns.ipynb:48-53). They are not verified generators of the
  selected unadjusted master and must not replace its contents.

The scan establishes snapshot contents and key agreement only. Observation
publication dates, historic release availability and generator execution
lineage remain unverified. No downstream check removes those limitations.

## Approved validation policy — D63

Retain original master fews_ipc_crisis. Accept observed phases only when finite,
integral and in 1..5, with binary 0 for phases 1/2 and 1 for 3/4/5. Check observed
master binary/phase agreement and exact same-key ledger consistency. Missing
observations remain missing; never derive persistence zero from missing phase.
Preserve the full master monthly grid and D36's exact-origin persistence rule.

Validate unique canonical area/month keys and ledger date-field consistency.
Exclude only the identified terminal one-field artifact in the pinned ledger's
parsed view, logging its source hash, physical line, raw value and reason.
Keep raw sources unchanged. Any other malformed record/key, duplicate key,
nonempty invalid phase/binary or master/ledger disagreement stops preflight
for investigation, without silent deduplication, deletion or relabeling.
Ordinary absent labels/observations remain supported missingness. This policy
is approved for planning only and does not authorize model execution.
