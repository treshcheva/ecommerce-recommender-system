# Executed full artifacts

These notebooks preserve the completed full experiment:

1. [02 — Modeling](02_modeling.ipynb)
2. [03 — Evaluation](03_evaluation.ipynb)
3. [04 — Learning to rank](04_learning_to_rank.ipynb)

See [the report](../REPORT.md), [metrics](../metrics/) and [300 dpi figures](../figures/).
The only editable notebook source is [notebooks/](../../../notebooks/README.md).
Future runners read that directory; they never execute these saved copies as
an alternative implementation.

During consolidation, all cell sources were synchronized with the canonical
notebooks without rerunning models. Original outputs/execution counts are
preserved; generated full conclusions are retained as display outputs.
Metadata explicitly records source synchronization without repeat execution.
Original byte-for-byte executed notebooks are archived in
[the source-unification backup](../logs/source_unification_before/full/).
Runtime metadata selects full mode and checkpoint reuse; it does not change
cell source. Source alignment and output preservation are checked by
`python -m scripts.check_notebook_sources`.
