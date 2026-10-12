# Local streaming D30 parity: pre-registered protocol (written before any local reduction was computed)

- **Science:** unchanged D30. The local reducer is `scripts/aef_local_d30.py`; its semantics are listed in its
  docstring.
- **References:** immutable saved EE responses only, with no new EE aggregation.
  - Selection: C02 = admin 3447 whole (pilot2 rev2, single source).
  - Held out:
    - C03 = the six 3447 lattice windows (pilot2 rev2);
    - C01 = the admin-1722 overlap piece (six candidates, cross-UTM first-valid);
    - admin 658 = whole-admin interactive response plus its two retained full-run windows.
- **Weight-variant selection:** the EE C02 valid area 51,234,473.3333 m² = 512,344.7333 cells, a fractional part of
  11/15, indicates 8-bit k/255 weights.
  - Variants: exact, round255, floor255, ceil255.
  - Exactly ONE is selected, using C02 only: the variant with the smallest |A relative difference|.
  - That variant is then fixed for every held-out reference.
- **Gates** (existing reuse-diagnostic tolerances; never relaxed after seeing outputs):
  - A relative ≤ 1e-7;
  - max relative S and Q ≤ 1e-6;
  - descriptor (128 mean/SD) max absolute ≤ 1e-6;
  - coverage absolute ≤ 1e-7, with the same usable route.
  - Each must hold on the selection reference AND on every held-out reference evaluated.
- **Count diagnostic:** the local centre-inclusion valid count against EE `valid_cells`. It is reported, not a gate.
- **Relative-difference scale:**
  - |x − ref| / max(|ref|, 1e-300) for A and per band.
  - Near-zero S bands are reported with absolute differences as well; the gate still uses the relative form.
  - Q is always positive, since the values are squares.
- **Stop rule:** if the selection reference or any held-out reference fails, the route is declared incompatible with
  D30 parity at the fixed tolerances. Optional downloads stop and the result is reported. Nothing is relaxed.
