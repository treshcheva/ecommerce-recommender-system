"""Candidate generation from the project's collaborative recommenders."""

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


_CANDIDATE_COLUMNS = [
    "user_id",
    "item_id",
    "als_score",
    "als_rank",
    "has_als",
    "itemknn_score",
    "itemknn_rank",
    "has_itemknn",
    "candidate_source",
]


def generate_candidate_frame(
    user_ids: Sequence[int],
    history_events: pd.DataFrame,
    als_model,
    itemknn_model,
    candidates_per_model: int = 100,
) -> pd.DataFrame:
    """Union top candidates from library ALS and ItemKNN, excluding seen items."""
    if candidates_per_model < 1:
        raise ValueError("candidates_per_model must be positive.")
    if "visitorid" not in history_events or "itemid" not in history_events:
        raise ValueError("history_events must contain visitorid and itemid columns.")

    seen_by_user = (
        history_events[history_events["visitorid"].isin(user_ids)]
        .groupby("visitorid")["itemid"]
        .agg(lambda values: set(int(value) for value in values))
        .to_dict()
    )
    rows = []
    for user_id in user_ids:
        seen = seen_by_user.get(user_id, set())
        by_item: Dict[int, Dict[str, object]] = {}
        for source, model in (
            ("als", als_model),
            ("itemknn", itemknn_model),
        ):
            ranked = model.recommend_with_scores(
                user_id, seen_items=seen, k=candidates_per_model
            )
            for rank, (item_id, score) in enumerate(ranked, start=1):
                item_id = int(item_id)
                if item_id in seen:
                    continue
                candidate = by_item.setdefault(
                    item_id,
                    {
                        "user_id": int(user_id),
                        "item_id": item_id,
                        "als_score": np.nan,
                        "als_rank": np.nan,
                        "has_als": False,
                        "itemknn_score": np.nan,
                        "itemknn_rank": np.nan,
                        "has_itemknn": False,
                    },
                )
                candidate["{}_score".format(source)] = float(score)
                candidate["{}_rank".format(source)] = rank
                candidate["has_{}".format(source)] = True

        for candidate in by_item.values():
            candidate["candidate_source"] = (
                "both"
                if candidate["has_als"] and candidate["has_itemknn"]
                else "als_only"
                if candidate["has_als"]
                else "itemknn_only"
            )
            rows.append(candidate)

    candidates = pd.DataFrame(rows, columns=_CANDIDATE_COLUMNS)
    if not candidates.empty and candidates.duplicated(["user_id", "item_id"]).any():
        raise RuntimeError("Candidate generation produced duplicate user-item pairs.")
    return candidates


def candidate_recall(
    candidates: pd.DataFrame,
    target_events: pd.DataFrame,
    history_events: pd.DataFrame,
    user_ids: Sequence[int],
    catalog_items: Iterable[int],
    k: int = 100,
) -> pd.DataFrame:
    """Report macro recall with explicit per-source and union budgets.

    ``recall`` is labeled by ``metric`` and ``cutoff`` in each row. The union
    uses only candidates in either source's top k, so its budget is 2*k even
    if the input contains deeper candidates. Missing users contribute zero.
    A multi-snapshot input must be sliced before calling this function.
    """
    if k < 1:
        raise ValueError("k must be positive.")
    if "snapshot_id" in candidates and candidates["snapshot_id"].nunique() > 1:
        raise ValueError("Select a single snapshot before measuring candidate recall.")
    if candidates.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Candidates must be unique by user_id and item_id.")
    catalog = set(int(item_id) for item_id in catalog_items)
    targets = target_events[
        target_events["visitorid"].isin(user_ids)
        & target_events["itemid"].isin(catalog)
    ]
    seen_by_user = (
        history_events[history_events["visitorid"].isin(user_ids)]
        .groupby("visitorid")["itemid"]
        .agg(lambda values: set(int(value) for value in values))
        .to_dict()
    )
    relevant_by_user = {}
    for user_id, group in targets.groupby("visitorid"):
        seen = seen_by_user.get(int(user_id), set())
        relevant = set(int(value) for value in group["itemid"]) - seen
        if relevant:
            relevant_by_user[int(user_id)] = relevant
    if not relevant_by_user:
        raise ValueError("No users have a new target item in the benchmark catalog.")

    generator_columns = {
        "implicit ALS": "als_rank",
        "ItemKNN": "itemknn_rank",
    }
    results = []
    for generator, rank_column in generator_columns.items():
        per_user = []
        for user_id, relevant in relevant_by_user.items():
            user_candidates = candidates[
                (candidates["user_id"] == user_id)
                & (candidates[rank_column] <= k)
            ]
            found = set(int(value) for value in user_candidates["item_id"])
            per_user.append(len(found & relevant) / len(relevant))
        results.append(
            {
                "generator": generator,
                "metric": "Recall@{}".format(k),
                "cutoff": k,
                "recall": float(np.mean(per_user)),
                "users": len(per_user),
            }
        )

    per_user_union = []
    for user_id, relevant in relevant_by_user.items():
        user_candidates = candidates[
            (candidates["user_id"] == user_id)
            & ((candidates["als_rank"] <= k) | (candidates["itemknn_rank"] <= k))
        ]
        found = set(int(value) for value in user_candidates["item_id"])
        per_user_union.append(len(found & relevant) / len(relevant))
    results.append(
        {
            "generator": "ItemKNN + implicit ALS (up to {} unique)".format(2 * k),
            "metric": "Recall@{}".format(2 * k),
            "cutoff": 2 * k,
            "recall": float(np.mean(per_user_union)),
            "users": len(per_user_union),
        }
    )
    return pd.DataFrame(results)


def candidate_recall_at_100(*args, **kwargs) -> pd.DataFrame:
    """Compatibility alias; returns the explicit cutoff/metric/recall schema."""
    return candidate_recall(*args, **kwargs)
