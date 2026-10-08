# Read-only enumeration, 2026-10-08

Using live accepted child view/evaluation_view.json own raw panels, exact whitelist in design.md:

| Family | Summary rows | External versions | Registry names |
|---|---:|---:|---:|
|P6|32|8|8|
|MLP|248|48|16|
|Yearly|96|12|12|
|Climate|32|8|8|
|Window|56|16|16|
|Split2024|28|8|8|
|Total|492|100|68|

Uniform9 keys: binary accuracy/precision/recall/f1/f2;four_class accuracy/macro_f1;q3_r2_projected;n.4,424 finite values,4 NA: window H6 new_local_support macroF1 for all four arms (class1 absent from truth and prediction). Do not fill missing values.

Current whitelist has no saved n=0 own panel. Split H12 main n0 is separately recorded but panels absent; no synthesized Summary row. Window new_local_support H3/H12 absent. Saved representation panels[namespace]={source_path,panel}; extract.py:88–103/import_runs.py:217–230. Other families use main/supplementary only; window uses selected_dates aggregates only.

100 external versions =76 fresh_trained+16 diagnostic_local+8 reused_comparator; MLP seeds as versions yield16 names. None are newly trained by registration. Counts are proposed and enumerated read-only, not yet created.
