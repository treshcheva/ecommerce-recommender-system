"""Offline ranking metrics for time-based implicit-feedback evaluation."""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.recommendation import DEFAULT_EVENT_WEIGHTS


def evaluate_recommender(
    recommender,
    target_events: pd.DataFrame,
    history_events: pd.DataFrame,
    user_ids: Sequence[int],
    catalog_items: Sequence[int],
    k: int = 10,
    relevance_mode: str = "binary",
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[int, List[int]]]:
    """Evaluate novel catalog recommendations for users with relevant targets.

    Binary target relevance is the default for all primary experiments. Optional
    graded relevance uses the maximum event strength per user-item pair.
    Previously observed items are removed using history available strictly
    before that window.
    """
    if k < 1:
        raise ValueError("k must be positive.")
    if relevance_mode not in {"graded", "binary"}:
        raise ValueError("relevance_mode must be 'graded' or 'binary'.")
    if not user_ids:
        raise ValueError("user_ids must not be empty.")

    catalog = set(int(item_id) for item_id in catalog_items)
    targets = target_events[
        target_events["visitorid"].isin(user_ids)
        & target_events["itemid"].isin(catalog)
    ].copy()
    unknown_events = sorted(set(targets["event"].unique()) - set(DEFAULT_EVENT_WEIGHTS))
    if unknown_events:
        raise ValueError("No target relevance configured for: {}".format(unknown_events))
    if relevance_mode == "binary":
        targets["relevance"] = 1.0
    else:
        targets["relevance"] = targets["event"].map(DEFAULT_EVENT_WEIGHTS)
    target_by_user = targets.groupby(["visitorid", "itemid"])["relevance"].max()
    targets_by_user = {
        int(user_id): user_targets
        for user_id, user_targets in target_by_user.groupby(level=0)
    }

    seen_by_user = (
        history_events[history_events["visitorid"].isin(user_ids)]
        .groupby("visitorid")["itemid"]
        .agg(lambda values: set(int(value) for value in values))
        .to_dict()
    )
    recommendations: Dict[int, List[int]] = {}
    rows = []
    discount = 1.0 / np.log2(np.arange(2, k + 2))

    for user_id in user_ids:
        seen = seen_by_user.get(user_id, set())
        target_items = targets_by_user.get(user_id)
        if target_items is None:
            continue
        relevant = {
            int(item_id): float(relevance)
            for (_, item_id), relevance in target_items.items()
            if int(item_id) not in seen
        }
        if not relevant:
            continue

        recs = recommender.recommend(user_id, seen_items=seen, k=k)
        recommendations[int(user_id)] = recs
        gains = np.asarray(
            [((2.0 ** relevant[item_id]) - 1.0) if item_id in relevant else 0.0
             for item_id in recs[:k]],
            dtype=np.float64,
        )
        dcg = float(np.sum(gains * discount[: len(gains)]))
        ideal_relevance = sorted(relevant.values(), reverse=True)[:k]
        ideal_gains = np.asarray(
            [(2.0 ** value) - 1.0 for value in ideal_relevance], dtype=np.float64
        )
        idcg = float(np.sum(ideal_gains * discount[: len(ideal_gains)]))
        hits = len(set(recs[:k]) & set(relevant))
        rows.append(
            {
                "user_id": int(user_id),
                "ndcg@{}".format(k): dcg / idcg if idcg else 0.0,
                "recall@{}".format(k): hits / len(relevant),
                "precision@{}".format(k): hits / k,
                "relevant_items": len(relevant),
                "hits": hits,
            }
        )

    per_user = pd.DataFrame(rows)
    if per_user.empty:
        raise ValueError(
            "No evaluation users have a novel target item in the benchmark catalog."
        )
    recommended_items = {
        item_id for ranked_items in recommendations.values() for item_id in ranked_items
    }
    summary = {
        "ndcg@{}".format(k): float(per_user["ndcg@{}".format(k)].mean()),
        "recall@{}".format(k): float(per_user["recall@{}".format(k)].mean()),
        "precision@{}".format(k): float(per_user["precision@{}".format(k)].mean()),
        "coverage@{}".format(k): (
            float(len(recommended_items) / len(catalog)) if catalog else 0.0
        ),
        "evaluated_users": int(len(per_user)),
        "catalog_items": int(len(catalog)),
        "recommended_unique_items": int(len(recommended_items)),
    }
    return summary, per_user, recommendations
