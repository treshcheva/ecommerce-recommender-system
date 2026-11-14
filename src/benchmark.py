"""Train-only benchmark profiles, artifact isolation, and cost preflight."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.recommendation import select_benchmark_users, select_catalog_items
from src.ranking_features import split_history_target


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    max_users: int
    max_items: int
    min_history: int = 5
    seed: int = 42
    k: int = 10
    candidates_per_model: int = 100
    relevance_mode: str = "binary"


PROFILES = {
    "dev": BenchmarkConfig("dev", max_users=20000, max_items=1500),
    "full": BenchmarkConfig("full", max_users=100000, max_items=10000),
}


def get_benchmark_config(mode=None):
    mode = mode or os.environ.get("BENCHMARK_MODE", "dev")
    if mode not in PROFILES:
        raise ValueError("BENCHMARK_MODE must be dev or full.")
    return PROFILES[mode]


def artifact_root(root: Path, config: BenchmarkConfig) -> Path:
    # Preserve established dev paths; expanded results can never overwrite them.
    return root / "outputs" if config.mode == "dev" else root / "outputs" / "full"


def benchmark_manifest(train, cohort, catalog, config):
    """Fingerprint the actual past data and ordered cohort/catalog for reuse."""
    columns = ["event_time", "visitorid", "itemid", "event"]
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(train[columns], index=False).values.tobytes()
    ).hexdigest()
    return {
        "config": asdict(config),
        "train_sha256": digest,
        "train_start": str(train.event_time.min()),
        "train_end": str(train.event_time.max()),
        "train_rows": len(train),
        "cohort_sha256": hashlib.sha256(json.dumps(cohort).encode()).hexdigest(),
        "catalog_sha256": hashlib.sha256(json.dumps(catalog).encode()).hexdigest(),
        "users": len(cohort), "items": len(catalog),
    }


def validate_selection(selected, manifest):
    if selected.get("benchmark") != manifest:
        raise ValueError("Selected parameters do not match this binary benchmark; rerun notebook 02.")
    if selected.get("relevance_mode") != "binary":
        raise ValueError("Primary parameters must be selected with binary relevance.")


def estimate_benchmark_cost(train: pd.DataFrame) -> pd.DataFrame:
    """Preflight both profiles without reading validation/test or fitting models.

    Counts and pair-work proxies are exact on train; memory numbers are planning
    bounds rather than measured RSS. Time ratios are proxies, not wall times.
    Include inner rank-training history because dev already uses all eligible
    users there, independently of its smaller evaluation cohort.
    """
    rank_history, rank_target, _ = split_history_target(train)
    rank_counts = rank_history.groupby("visitorid").size()
    rank_users = rank_counts[rank_counts >= 5].index
    rows = []
    for config in PROFILES.values():
        cohort = select_benchmark_users(train, config.min_history, config.max_users, config.seed)
        catalog = select_catalog_items(train, config.max_items)
        events = train[train.visitorid.isin(cohort) & train.itemid.isin(catalog)]
        pairs = events[["visitorid", "itemid"]].drop_duplicates()
        degree = pairs.groupby("visitorid").size()
        rank_catalog = select_catalog_items(rank_history, config.max_items)
        rank_events = rank_history[
            rank_history.visitorid.isin(rank_users) & rank_history.itemid.isin(rank_catalog)
        ]
        rank_pairs = rank_events[["visitorid", "itemid"]].drop_duplicates()
        rank_degree = rank_pairs.groupby("visitorid").size()
        rank_candidate_users = rank_target[
            rank_target.visitorid.isin(rank_users) & rank_target.itemid.isin(rank_catalog)
        ].visitorid.nunique()
        max_candidate_rows = 2 * config.candidates_per_model * rank_candidate_users
        rows.append({
            "mode": config.mode, "users": len(cohort), "items": len(catalog),
            "train_events": len(events), "train_nnz": len(pairs),
            "itemknn_pair_work": int((degree ** 2).sum()),
            "rank_train_users": len(rank_users), "rank_train_nnz": len(rank_pairs),
            "rank_itemknn_pair_work": int((rank_degree ** 2).sum()),
            "rank_candidate_users": rank_candidate_users,
            "rank_candidate_rows_upper": max_candidate_rows,
            # Sparse product could approach dense; CSR + Python LIL + copies.
            "itemknn_legacy_dense_allowance_mb": 80 * max(len(catalog), len(rank_catalog)) ** 2 / 2**20,
            # Two CSR block workspaces, final top-100 CSR, both orientations
            # of input, plus factor-of-two workspace margin. No dense I²/LIL.
            "itemknn_planning_mb": 2 * (
                2 * 128 * max(len(catalog), len(rank_catalog)) * 8
                + max(len(catalog), len(rank_catalog)) * 100 * 8
                + 2 * max(len(pairs), len(rank_pairs)) * 8
            ) / 2**20,
            "rank_features_planning_mb": max_candidate_rows * 1024 / 2**20,
            "als_factor_arrays_mb": (len(cohort) + len(catalog)) * 64 * 4 / 2**20,
            "evaluation_score_work_upper": len(cohort) * len(catalog),
        })
    result = pd.DataFrame(rows)
    for column in ["train_nnz", "itemknn_pair_work", "rank_itemknn_pair_work",
                   "rank_candidate_rows_upper", "evaluation_score_work_upper"]:
        result[column + "_vs_dev"] = result[column] / max(result.iloc[0][column], 1)
    return result
