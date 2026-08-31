# Ethiopia Forecasting Experiment

This context defines the language used to distinguish baseline forecasting,
future-weather information, and evidence strength in the Ethiopia study.

## Language

**Experiment cohort**:
The Ethiopia FEWS NET administrative units included in an experiment run.
_Avoid_: Reference CDS areas, Ethiopia bounding-box sample

**Forecast origin month**:
The month at which all information available to a forecast is frozen.
_Avoid_: Current month, feature month

**Valid month**:
The month for which a weather forecast value applies.
_Avoid_: Forecast date

**Target month**:
The month whose food-crisis outcome is predicted.
_Avoid_: Test date

**Forecast horizon**:
The number of calendar months from the forecast origin month to the target
month.
_Avoid_: Scope number, generic lag

**Forecast lead**:
The number of calendar months from the forecast origin month to a weather
forecast's valid month. Lead 0 is the origin month.
_Avoid_: Target lag

**Target-relative lag**:
The number of calendar months from a weather value's valid month to the target
month. For horizon `H` and forecast lead `j`, it equals `H - j`.
_Avoid_: Lead

**Inclusive lead-0-to-6 path**:
Seven monthly weather values covering the origin month and the following six
valid months.
_Avoid_: Six-value path

**Frozen baseline**:
The reviewed Ethiopia forecasting configuration against which an experimental
change is compared.
_Avoid_: Original global result, current code defaults

**CDS-enhanced forecast**:
A forecast that differs from its matched frozen baseline only through approved
CDS-derived weather information and its documented transformations.
_Avoid_: New baseline

**Forward prediction comparison**:
A comparison of predicted classes or probabilities before target outcomes are
available.
_Avoid_: Performance evaluation, accuracy improvement

**Delayed outcome evaluation**:
A performance assessment performed after the corresponding FEWS NET target
labels become available.
_Avoid_: Immediate validation

**Overfitting hypothesis**:
A concern that the baseline does not generalize across held-out time or space;
it becomes a finding only after diagnostic evidence supports it.
_Avoid_: Proven overfitting

**Nowcasting ceiling**:
The advisor-proposed upper information bound in which the model receives
perfect target-period weather information. It is a theoretical comparison
boundary, not an observed performance result.
_Avoid_: Guaranteed CDS performance

