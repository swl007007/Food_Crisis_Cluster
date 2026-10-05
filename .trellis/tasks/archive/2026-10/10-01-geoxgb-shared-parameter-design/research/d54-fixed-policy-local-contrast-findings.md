# D54: zero-fit fixed-policy local-increment contrast (2026-10-02)

**Scope:** contract `d54-fixed-policy-local-contrast-plan.md`, planning commit `c92a4c3`. The primary crisis-F1 endpoint, four-class main contract and final criterion are unchanged.
- **Change:** D38's already-fixed post-hoc prior q (origin class .625, others .125; missing origin unchanged) applied to both the saved D34 root and the saved full Brier-local probabilities (D41 rows, frozen maps and routes). Zero fits, model loads or free parameters.
- **Arms:** raw root, raw full, post root, post full; persistence on matched exact-origin keys only.
- **Interpretation limits:** the interaction was chosen after exposed development results, so it is not pre-registered. The transform is the same prior reweighting with no tree change; only full-vs-root log-odds differences are preserved, probability differences may change. Post-hoc outputs are diagnostic; this is not partition training on an anchored base and not evidence that such training would work.

## Producer and run (executor factual record, `16ee1f4`)

- **Producer** `6d5619b`: `research/d54_fixed_policy_local_contrast.py` (652 lines, numpy/pandas/stdlib only), git blob `81ea62ee76092e86c7ee32c3256f9909fa260890`, sha256 `3b356631a6e901ce2179862e9cc61fbd010579e611a36e4b7500d240eb62b55b`. The native check made `reconcile_d39` fail closed on a missing record and added selftest coverage.
- **Run:** `C:\Users\swl00\geoxgb_runs\geoxgb-d54-fixed-policy-local-contrast-20261002`, frozen Windows Python 3.12.10 with assertions on (`sys.flags.optimize = 0`; command, exit and runtime in `research/d54-run.log`), exit 0 in 6 s.
  - **Gates:** all passed — D41/D52 key, probability and truth alignment; D52 origin codes; missing-origin neutrality; zero-increment identity.
  - **Reconciliation:** D41 confusions (all / matched / per root / per H / routes, C and E3) exact; D38 E3 post(root) probabilities with maximum difference 0.0 over 113,508 rows; D39 per-root post-hoc confusions exact.
  - **Rows:** C 207,539 (78,017 missing origin); E3 113,508 (713 missing origin).

## Verification (supervisor, distinct from the executor record)

`research/d54_supervisor_check.py` → `research/d54_supervisor_results.json`, log `research/d54-supervisor.log`: **PASS**.
- 8,454 checks: 378 per-root cells, 16 pooled cells, 326 change cells, 16 change totals, 64 input hashes, 4 output hashes; D38 post-root maximum difference 0.0; no producer import, model load or fit.
- The verifier writes its outputs beside itself. Its canonical rerun path is the external copy, `C:\Users\swl00\geoxgb_runs\d54-supervisor-verification\verify_d54.py`; the task-research copy is evidence only.

## Results (E3, matched exact-origin keys)

| H | Crisis F1: raw root / raw full / post root / post full / persistence | Mean-fold: post root / post full / persistence |
|---|---|---|
| 4 | .629050 / .629398 / .652273 / .652273 / .651697 | .655916 / .655916 / .655255 |
| 8 | .533375 / .533323 / .555135 / .554766 / .555614 | .553086 / .552701 / .555568 |
| 12 | .478355 / .479874 / .540797 / .543346 / .549808 | .537615 / .540748 / .547711 |
| All 21 | .5515156652 / .5521789152 / .5839738224 / .5846255194 / .5864055300 | .5822054687 / .5831214943 / .5861777298 |

- **Fold wins (wins / ties / losses), post full vs post root:** H4 0/7/0, H8 4/1/2, H12 6/0/1; all 21 10/8/3. **Post full vs persistence:** H4 1/6/0, H8 4/0/3, H12 3/0/4; all 21 8/6/7. Raw full vs raw root: all 21 13/0/8. Not significance tests.
- **E3 local − root changes (local routes):**

  | Pair | Δ TP | Δ FP | Δ FN | Crisis flips |
  |---|---|---|---|---|
  | raw full − raw root | +78 | +162 | −78 | 580 |
  | post full − post root | +26 | +19 | −26 | 103 (H4 0, H8 34, H12 69) |

  Zero-increment rows: no change. All-key and matched E3 changes are identical.
- **Normalised crisis Brier (all 21):** post root .1159761348 → post full .1160490068; raw root .1018845949 → raw full .1019450125; persistence one-hot .1464071989. Per H, post full vs post root: H4 .104444 / .104471, H8 .124607 / .124397, H12 .119140 / .119104.

**C (in-window interpolation; context only).** Matched F1, raw root / raw full / post root / post full / persistence: H4 .715605 / .718702 / .645206 / .645631 / .642024; H8 .727744 / .733961 / .580367 / .581370 / .495687; H12 .589058 / .607705 / .462318 / .468489 / .415537. Local − root changes on C local routes, by key set (denominators differ):

| C key set | raw Δ TP / Δ FP | post Δ TP / Δ FP |
|---|---|---|
| All keys (147,305 local rows) | +414 / +197 | +232 / +116 |
| Matched exact-origin keys | +249 / +125 | +67 / +44 |

## Supervisor decision

**D54 is complete; no new prediction policy is adopted.**
- At the fixed D38 operating point the existing full-local increment offers a tiny, heterogeneous E3 benefit: pooled post root .5839738224 → post full .5846255194 against persistence .5864055300; mean-fold .5822054687 → .5831214943 against .5861777298.
- H4: no decision changes. H8: negative pooled and mean-fold. H12: positive but still below persistence.
- Fold counts 10/8/3 (vs post root) and 8/6/7 (vs persistence) are descriptive, not significance tests.
- E3 post changes +26 TP / +19 FP; normalised crisis Brier slightly worsens (.1159761348 → .1160490068).
- The stronger raw gains on C do not establish forward transfer; C all-key and matched changes are reported separately and not mixed.
- See `research/stage1-research-decision-d54.md` for the Stage 1 research decision.

## Evidence

**Task research holds exact byte copies:**

| File | Bytes | sha256 |
|---|---|---|
| `d54_summary.json` | 36,586 | `a014515b237814d6b36d48ebffb7226187d7aad76409f8e8b6739b52fcd36759` |
| `d54_per_root.csv` | 47,598 | `5d999f802447d037d7705fdb0f926aabea3ccdee9e85fa7443100324e35c91ea` |
| `d54_changes.csv` | 29,922 | `c2fa09290f50ab4d011a69d75b3da45f89870427ceef78d2ea95320ff7f08dee` |
| `d54_identity.json` | 9,179 | `a422eb02eddb664cd596af5c9abf9b8de4c2b4339bc16fa3c800d16d32b04c02` |
| `d54_completion.json` | 407 | `abe1f6f261c0d1f1b938964d9ab92fb5afafb2444917b97e96fa177d962f4b4a` |
| `d54-run.log` | 375 | `05a32f4b09c816b581af939cc956f4162ce78e4c4c8fefb265b260d174538c17` |
| `d54_supervisor_check.py` | 19,888 | `16102eeac615635ad3538289e6b858eca53feb6d35d3299a9ef082dc069754ff` |
| `d54_supervisor_results.json` | 11,919 | `a539a69bf77f21777ce7e41b426119c895e236191e150ece959e9633594046ab` |
| `d54-supervisor.log` | 755 | `23ec437cc8152fe2a09f8615c6a51b4234fbb4a0271af75c6c4ad617930a745c` |
