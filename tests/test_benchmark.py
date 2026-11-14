import pandas as pd
import pytest

from src.benchmark import (
    artifact_root, benchmark_manifest, estimate_benchmark_cost,
    get_benchmark_config, validate_selection,
)
from src.recommendation import select_benchmark_users, select_catalog_items


def _train():
    return pd.DataFrame({
        "visitorid": [1] * 10 + [2] * 10,
        "itemid": [10, 20] * 10,
        "event": ["view"] * 20,
        "event_time": pd.date_range("2020-01-01", periods=20, tz="UTC"),
    })


def test_profiles_are_train_only_reproducible_and_outputs_are_isolated(tmp_path):
    train = _train()
    dev, full = get_benchmark_config("dev"), get_benchmark_config("full")
    assert full.max_items > dev.max_items
    assert full.max_users > dev.max_users
    users = select_benchmark_users(train, dev.min_history, dev.max_users, dev.seed)
    catalog = select_catalog_items(train, dev.max_items)
    manifest = benchmark_manifest(train, users, catalog, dev)
    assert manifest == benchmark_manifest(train.copy(), users, catalog, dev)
    selection = {"benchmark": manifest, "relevance_mode": "binary"}
    validate_selection(selection, manifest)
    with pytest.raises(ValueError, match="do not match"):
        validate_selection(selection, benchmark_manifest(train, users, catalog, full))
    changed = train.copy()
    changed.loc[0, "event"] = "transaction"
    with pytest.raises(ValueError, match="do not match"):
        validate_selection(selection, benchmark_manifest(changed, users, catalog, dev))
    assert artifact_root(tmp_path, dev) == tmp_path / "outputs"
    assert artifact_root(tmp_path, full) == tmp_path / "outputs/full"


def test_preflight_needs_only_train_and_records_explicit_budget():
    estimate = estimate_benchmark_cost(_train()).set_index("mode")
    assert estimate.loc["dev", "rank_candidate_rows_upper"] == 200
    assert estimate.loc["dev", "train_nnz"] == 4
    assert estimate.loc["dev", "train_nnz_vs_dev"] == 1
    assert (estimate["itemknn_planning_mb"] > 0).all()
    with pytest.raises(ValueError, match="dev or full"):
        get_benchmark_config("other")
