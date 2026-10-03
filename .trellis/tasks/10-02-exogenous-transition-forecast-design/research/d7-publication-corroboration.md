# D7: corroboration of FEWS NET report publication dates (bounded, ~10 minutes)

2026-10-02/03, Claude executor. Read-only HTTP requests. No model fit, no product or test edit, no 2025 outcome values. ReliefWeb was not retried (v1 returns 410; v2 needs an approved appname). FEWSNET.csv is treated as a directly downloaded original (user clarification), so no local generator was sought.

## Facts obtained

### Official report pages (fews.net, current site)

Fields extracted from each page's own metadata block:

| Page | `publicationDate` | Report type (page label) | Period (page label) |
|---|---|---|---|
| https://fews.net/east-africa/ethiopia/food-security-outlook/february-2020 | 2020-02-06 | Food Security Outlook | "February - September 2020" (the page body also says "February to September 2020") |
| https://fews.net/east-africa/ethiopia/food-security-outlook/october-2020 | 2020-10-06 | Food Security Outlook | "October 2020 - May 2021" |
| https://fews.net/east-africa/kenya/food-security-outlook/february-2020 | 2020-02-28 | Food Security Outlook | "February - September 2020" |

The raw field shape is `"publicationDate":{"type":"datetime","value":[{"value":"2020-02-06"}]}`. The page HTML contains no plain PDF link (the document link is apparently rendered client-side), so no PDF creation or modification date was obtained.

### Independent archive (Internet Archive CDX, earliest captures)

| Page | Earliest capture | Relation to the stated date |
|---|---|---|
| Ethiopia October 2020 | 2020-11-07 (HTTP 200) | after 2020-10-06: consistent, but only an upper bound |
| Kenya February 2020 | 2020-05-03 (HTTP 200) | after 2020-02-28: consistent, but only an upper bound |
| Ethiopia February 2020, February 2023 | not obtained (the Archive returned "Temporarily Offline", then timed out) | none |

## What this does and does not establish

- **Product and cycle linkage on the page (partly supported).** Each page is a Food Security Outlook whose label names the outlook cycle (February → February–September; October → October–May). This links the stated date to a named outlook cycle, and so plausibly to the Current Situation classification of that cycle month.
- **Not established:**
  - That FEWSNET.csv's row month (e.g. 2020-02) is exactly that report's CS. The CSV carries no document identifier; its own column definitions describe phases, not documents.
  - That the CS data, as opposed to the narrative report, was public on the page date.
- **Date corroboration (not achieved).** The archive captures are later than the stated dates. They are consistent with them but cannot confirm them, because a page could have been published at any time before its first capture. The Ethiopia February 2020 date (the one flagged for corroboration) has no independent capture. These are current, migrated pages, so the field may not be the original posting date.
- **Variation within one cycle.** Ethiopia (2020-02-06) and Kenya (2020-02-28) differ by 22 days for the same February 2020 cycle. Both fall within month M for these two pages, but neither this nor the FDW `created` evidence (which shows M+1 or later for 2021–2023 Feb cycles in four countries) supports any blanket M+0, M+1 or M+2 rule.

## Admissibility

**Not admissible** as a verified vintage, and not sufficient for a release rule: the dates are uncorroborated, the CSV-row linkage is unproven, and there are only three pages from two countries.

At most, these are source-labelled publication dates usable as candidate inputs to a disclosed retrospective reconstruction, and only after both of the following:
1. Linkage evidence ties each FEWSNET.csv cycle row to the report/product (or its CS dataset) carrying the date.
2. Coverage is enumerated for every country and cycle used in fitting, with missing cases kept explicit.

D7 remains **blocked** on the historical IPC release rule.

## Minimum decisive next facts (not done in this bound)

1. For a few historical (≤ 2024, unprotected) Ethiopia and Kenya cycles, fetch the FDW `ipcphase` records (CS, matching reporting month) and their `datasourcedocument`/document ids. Check whether the document is the same Food Security Outlook whose page carries the `publicationDate`, and whether FDW values for that reporting month match FEWSNET.csv's row month on a sample of areas. The second check establishes the CSV cycle mapping without any 2025 value.
2. Obtain an independent dated copy of the same report (an archived PDF, or an earlier archive capture once the Archive is back) for at least the Ethiopia February 2020 page.
