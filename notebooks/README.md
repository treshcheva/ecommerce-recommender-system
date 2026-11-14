# Canonical experiment notebooks

This directory is the only source for notebook code and research narrative.
The same cells serve `dev` and `full`. Execution outputs are stored separately;
edit these sources rather than files under `outputs/`.

1. [01_eda.ipynb](01_eda.ipynb) — distributions, sparsity and temporal splits.
2. [02_modeling.ipynb](02_modeling.ipynb) — popularity baselines, ItemKNN/ALS,
   validation-only hyperparameter searches and paired bootstrap.
3. [03_evaluation.ipynb](03_evaluation.ipynb) — fixed-model test evaluation,
   uncertainty, errors, cold-start and recommendation examples.
4. [04_learning_to_rank.ipynb](04_learning_to_rank.ipynb) — CatBoost search,
   complete feature ablations, final comparison and conditional confidence intervals.

Executed tables, plots and conclusions are visible in the
[dev artifacts](../outputs/dev/notebooks/README.md) and
[full artifacts](../outputs/full/notebooks/README.md).
Existing outputs and result files were preserved during consolidation;
experiments were not rerun to synchronize source code.

```bash
python -m pip install -r requirements.txt
python -m scripts.run_eda
python -m scripts.run_experiments --mode dev
python -m scripts.run_experiments --mode full --estimate-only
python -m scripts.run_full
python -m scripts.check_notebook_sources
```

The runners read `notebooks/` and save executed copies to `outputs/dev/notebooks/`
(dev) or `outputs/full/notebooks/` (full). The source files remain unchanged.
They pass mode/checkpoint policy through `BENCHMARK_MODE` and
`RESUME_EXPERIMENTS`; they never rewrite cell code. Both modes use the same
300 dpi plotting and memory cleanup. Interactive execution defaults to dev
and a fresh search; to preserve clean sources, use the runners for 02–04.
Notebook 01 can be executed into a separate directory:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb --output-dir outputs/dev/notebooks
```

Primary selection/evaluation uses binary relevance. Training weights are 1/3/5.
Grids remain directly in 02/04 and test never selects parameters. Dev caps are
20,000 users / 1,500 items; full caps are 100,000 / 10,000. Artifact directories
isolate mode-specific parameters, metrics, figures and models.

The full runner skips compatible completed notebooks and enables checkpoint
reuse for an unfinished stage (`RESUME_EXPERIMENTS=1`). Fresh dev runs set it to
0, including reproducibility checks. Changed canonical code or damaged saved
notebooks are detected before a completed stage can be silently skipped.
The current full experiment is complete; no training is required to read it.
