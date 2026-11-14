# Executed experiment notebooks

Run the narrative notebooks in order:

1. [01_eda.ipynb](01_eda.ipynb) loads the raw data, studies distributions and
   sparsity, and checks the chronological split.
2. [02_modeling.ipynb](02_modeling.ipynb) fits Global/Recent Popularity,
   ItemKNN, and library ALS; its small ItemKNN/ALS searches use validation only.
3. [03_evaluation.ipynb](03_evaluation.ipynb) loads the selected parameters,
   refits the four baseline/recommender models on train + validation, and
   evaluates once on test.
4. [04_learning_to_rank.ipynb](04_learning_to_rank.ipynb) tunes CatBoostRanker
   and feature ablations on validation, then compares five models on the
   held-out test and analyzes quality by user activity.

The notebooks are saved with executed outputs so their explanations, tables,
and inline figures are visible on GitHub. They import reusable logic from
`src/`; generated plots and metric CSVs are also saved in `outputs/`.

Install dependencies and generate `data/processed/` before running:

```bash
python -m pip install -r requirements.txt
python -m scripts.run_eda
jupyter lab
```
