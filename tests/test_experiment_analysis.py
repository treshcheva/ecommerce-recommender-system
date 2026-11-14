import pandas as pd
import pytest

from src.experiment_analysis import paired_metric_comparison


def test_bootstrap_pairs_users_by_id():
    frames = {
        "base": pd.DataFrame({"user_id": [1, 2, 3], "ndcg@10": [0.1, 0.4, 0.7]}),
        "model": pd.DataFrame({"user_id": [3, 1, 2], "ndcg@10": [0.8, 0.2, 0.5]}),
    }
    result = paired_metric_comparison(frames, "base", n_bootstrap=200).iloc[0]
    assert result.delta == pytest.approx(0.1)
    assert result.ci_low == pytest.approx(0.1)
    assert result.ci_high == pytest.approx(0.1)


def test_reject_mismatched_cohorts():
    frames = {
        "base": pd.DataFrame({"user_id": [1], "ndcg@10": [0.1]}),
        "model": pd.DataFrame({"user_id": [2], "ndcg@10": [0.2]}),
    }
    with pytest.raises(ValueError, match="identical"):
        paired_metric_comparison(frames, "base")


def test_bootstrap_is_reproducible_and_keeps_uncertainty_metadata():
    frames = {
        "base": pd.DataFrame({"user_id": [1, 2, 3], "ndcg@10": [0.1, 0.4, 0.7]}),
        "model": pd.DataFrame({"user_id": [1, 2, 3], "ndcg@10": [0.3, 0.2, 0.9]}),
    }
    first = paired_metric_comparison(frames, "base", seed=17, n_bootstrap=300)
    second = paired_metric_comparison(frames, "base", seed=17, n_bootstrap=300)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0].ci_low < 0 < first.iloc[0].ci_high
    assert first.iloc[0]["seed"] == 17
    assert first.iloc[0]["n_bootstrap"] == 300


def test_bootstrap_rejects_duplicate_users_and_missing_scores():
    import numpy as np
    base = pd.DataFrame({"user_id": [1, 2], "ndcg@10": [0.1, 0.2]})
    with pytest.raises(ValueError, match="unique"):
        paired_metric_comparison({"base": base, "bad": pd.concat([base, base])}, "base")
    with pytest.raises(ValueError, match="finite"):
        paired_metric_comparison({"base": base, "bad": base.assign(**{"ndcg@10": np.nan})}, "base")
