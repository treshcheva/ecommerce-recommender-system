# Executed dev artifacts

Read these notebooks for the saved dev tables and figures:

1. [01 — EDA](01_eda.ipynb)
2. [02 — Modeling](02_modeling.ipynb)
3. [03 — Evaluation](03_evaluation.ipynb)
4. [04 — Learning to rank](04_learning_to_rank.ipynb)

Edit [the canonical sources](../../../notebooks/README.md), not these artifacts.
Future CLI runs execute only those sources. Historical dev outputs and
execution counts were preserved while sources were synchronized; no experiment
was repeated during consolidation. Metadata records this distinction, and the
original versions are archived in `../../full/logs/source_unification_before/main/`.

Run `python -m scripts.run_experiments --mode dev` from the project root to
regenerate 02–04 here. Metrics/figures/models retain their established paths
under `outputs/`. Graded older research remains in `outputs/archive/`.
