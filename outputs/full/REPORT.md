# Completed full experiment

Cohort: 57,811 users; catalog: 10,000 items.
Events: train 370,990, validation 13,460, test 7,081.
Evaluated users: validation 1,627, test 1,058.

## Test metrics

| Model | NDCG@10 | Recall@10 | Precision@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| CatBoostRanker (ALS + ItemKNN) | 0.04455 | 0.06686 | 0.01295 | 0.34090 |
| ItemKNN | 0.03599 | 0.05348 | 0.01068 | 0.42160 |
| implicit ALS | 0.03168 | 0.04821 | 0.00907 | 0.25210 |
| Global Popularity | 0.00639 | 0.00597 | 0.00236 | 0.00340 |
| Recent Popularity (14d) | 0.00554 | 0.00929 | 0.00302 | 0.00240 |

## Paired bootstrap, full test

- CatBoost minus implicit ALS: delta +0.01287, 95% CI [+0.00759, +0.01852]; positive interval; supports higher NDCG on this cohort.
- CatBoost minus ItemKNN: delta +0.00856, 95% CI [+0.00282, +0.01424]; positive interval; supports higher NDCG on this cohort.

## Resources

- 02_modeling: 172.7 s.
- 03_evaluation: 24.3 s.
- 04_learning_to_rank: 171.6 s.
- Successful notebook total: 368.5 s.
- Interrupted first 04 attempt: 214.1 s to stop.
- Active runner time including both attempts/preflight, sampled: 590.3 s.
- First attempt stopped on swap growth; user authorized continuation without that threshold. Cached search/ablations reused; selected model reconstruction matched saved validation scores.
- Sampled peak runner + kernel RSS: 1.103 GiB (sum, shared pages may be double counted).
- Resource monitoring interval: 250 ms; sampling may miss shorter peaks.
- Models/grids and full benchmark unchanged. All PNG figures exported at 300 dpi.
- Dev notebooks, metrics, figures and models verified unchanged by SHA-256.

## Limits

- Dev/full use different cohorts and catalogs: raw score differences are not a controlled model improvement.
- Binary implicit events include views, not only purchases; only known eligible users/new catalog targets are evaluated.
- Single temporal split and training seed; percentile user bootstrap is conditional on fixed models and ignores shared-item dependencies.
- Validation search/early stopping/ablations share validation and produce optimistic diagnostic intervals; intervals are not multiplicity adjusted.

## Notebook source consolidation

The experiment above was not repeated. `notebooks/` is now the only canonical
source for dev/full. All cell sources (code and narrative) match the saved
artifacts exactly; run-specific conclusions are display outputs. Existing full
outputs, execution counts, metrics, figures and models were preserved.
Pre-consolidation notebook bytes remain in `logs/source_unification_before/`;
`logs/notebook_source_alignment.json` records the static parity/preservation
audit. Execution metadata distinguishes historical results from later source
synchronization. Future runners read only `notebooks/` and save executed
copies separately.
