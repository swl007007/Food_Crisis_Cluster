# D7 official website metadata follow-up

2026-10-02, Codex. Bounded read-only HTTP checks; no model fit, source rewrite or 2025 outcome-table access. This extends the earlier probe, which had not opened FEWS NET report pages. It does not certify a historical release ledger.

## Historical report metadata

The official report HTML contains a `publicationDate` field inside its page metadata. Direct requests returned:

| Requested report URL | Result | `publicationDate` |
|---|---|---|
| https://fews.net/east-africa/ethiopia/food-security-outlook/february-2020 | HTTP 200 | 2020-02-06 |
| https://fews.net/east-africa/ethiopia/food-security-outlook/october-2020 | HTTP 200 | 2020-10-06 |
| https://fews.net/east-africa/ethiopia/food-security-outlook/february-2023 | HTTP 200 | 2023-02-02 |
| https://fews.net/east-africa/kenya/food-security-outlook/february-2020 | HTTP 200 | 2020-02-28 |
| https://fews.net/east-africa/ethiopia/food-security-outlook/february-2018 | HTTP 404 | unavailable at this guessed URL |

Observed field shape: `"publicationDate":{"type":"datetime","value":[{"value":"2020-02-28"}]}`. This is a useful source-labelled date, distinct from FDW database `created`. It needs verification against the dated document/archive and a link from that report/product to the CS records before admission. In particular, the early Ethiopia dates need corroboration: a current migrated page is not itself a historical vintage, and report content may have been revised. Do not extrapolate these four dates into an all-country M+0 release rule or treat a guessed-URL 404 as an absent publication.

## Service-resumption announcement

Direct HTTP 200 at https://fews.net/fews-net-relaunches-website-resumes-global-food-security-analysis confirms the provider's statements:

> Following a brief pause in services during a review of U.S. foreign assistance programs, FEWS NET has resumed operations and published a new Global Food Security Update with analysis through September 2025.

> Regular monthly food security reporting for all FEWS NET-monitored countries will resume in the coming months, beginning with the publication of Key Messages for select countries in July 2025.

The page also says teams will work to fill data-collection gaps from the pause. No `publicationDate` field was found by the same extraction on this announcement. The earlier proposed exact June 24 posting date is therefore not verified here. Global updates and Key Messages are not automatically the country CS/ML1/ML2 products required by this model's ledger.

## Next evidence needed

Corroborate the historical report dates and product linkage, then enumerate country-cycle coverage and explicit missing cases. Obtain country/product-specific 2025 availability rather than inferring it from the global announcement. D7 real-fitting status remains **blocked**; no availability rule has changed.

A bounded attempt to corroborate historical posting dates through ReliefWeb obtained no report metadata: `/v1/reports` returned HTTP 410 (v1 decommissioned, use v2); `/v2/reports` returned HTTP 403 requiring an approved appname. The public updates search returned HTTP 202 without report results. These access outcomes do not establish any publication date.
