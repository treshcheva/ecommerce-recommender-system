"""Paired uncertainty estimates for fixed models on a shared evaluation cohort."""

from typing import Mapping

import numpy as np
import pandas as pd


def paired_metric_comparison(
    per_user: Mapping[str, pd.DataFrame],
    reference: str,
    metric: str = "ndcg@10",
    seed: int = 42,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Return model-minus-reference differences and percentile 95% intervals.

    Resample paired users, conditional on fixed trained models and this period.
    Require identical cohorts instead of silently dropping unmatched users.
    These intervals do not correct for model selection or temporal dependence.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if reference not in per_user:
        raise ValueError("reference must be present in per_user.")
    scores = {}
    for name, frame in per_user.items():
        if frame.empty or frame.user_id.duplicated().any():
            raise ValueError("Each model needs nonempty, unique user rows.")
        values = frame.set_index("user_id")[metric].sort_index()
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("Metrics must be finite.")
        scores[name] = values
    baseline = scores[reference]
    rows = []
    for name, values in scores.items():
        if not values.index.equals(baseline.index):
            raise ValueError("Models must have identical evaluation cohorts.")
        if name == reference:
            continue
        delta = (values - baseline).to_numpy()
        rng = np.random.default_rng(seed)
        # Bound memory even for a much larger evaluation cohort.
        means = np.empty(n_bootstrap)
        for start in range(0, n_bootstrap, 100):
            size = min(100, n_bootstrap - start)
            indices = rng.integers(0, len(delta), size=(size, len(delta)))
            means[start:start + size] = delta[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        rows.append({
            "model": name, "reference": reference, "users": len(delta),
            "metric": metric, "confidence": 0.95,
            "n_bootstrap": n_bootstrap, "seed": seed,
            "delta": float(delta.mean()), "ci_low": float(low),
            "ci_high": float(high),
            "relative_delta_pct": (
                float(100 * delta.mean() / baseline.mean())
                if baseline.mean() != 0 else np.nan
            ),
        })
    return pd.DataFrame(rows)
