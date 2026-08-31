# Ethiopia Forecasting Experiment

## Status

Preparation and contract-design only. No CDS download, data mutation, feature
generation, model fitting, or result regeneration has been authorized or
performed in this workspace.

## Purpose

This isolated workspace supports an Ethiopia-only forecasting study with two
ordered workstreams:

1. diagnose and improve the current baseline, with particular attention to
   overfitting and forecast-time feature availability;
2. only after the baseline is reviewed and frozen, evaluate a CDS-weather
   extension under the agreed forward-comparison contract.

The sequencing follows the advisor discussion recorded on 2026-08-31: even a
perfect future-weather path cannot theoretically provide information beyond a
weather-informed nowcasting ceiling, so unresolved baseline weaknesses must be
addressed before attributing value to CDS forecasts.

## Frozen boundaries

- Geography: Ethiopia only.
- Canonical spatial key: `FEWSNET_admin_code`.
- Cohort derivation: select `ISO3 == "ETH"` in the authoritative raw FEWSNET
  panel and freeze the resulting ID set.
- The `area_id` values in the reference CDS table are not experiment join keys.
- Production code, archived results, paper artifacts, and source data remain
  read-only unless a later implementation plan explicitly authorizes changes.
- The existing production horizons remain 4, 8, and 12 months.

## Workstream order

```text
Evidence inventory
  -> baseline overfitting and availability audit
  -> approved baseline improvement design
  -> frozen Ethiopia baseline
  -> formal CDS acquisition and harmonization design
  -> CDS-enhanced forward comparison
  -> delayed outcome evaluation when labels become available
```

## Workspace layout

```text
EthiopiaForecastingExperiment/
├── README.md
├── CONTEXT.md
├── baseline/
│   └── README.md
├── cds_weather/
│   └── README.md
├── docs/
│   ├── contract-draft.md
│   └── remaining-plan.md
└── evidence/
    └── 2026-08-31-initial-evidence.md
```

Large inputs, downloaded CDS files, generated features, model checkpoints, and
results do not belong in this preparation-only layout. Their storage and
manifest rules remain open design decisions.

The current task is closed with remaining work recorded in
[`docs/remaining-plan.md`](docs/remaining-plan.md). Execution requires a new,
explicit authorization.
