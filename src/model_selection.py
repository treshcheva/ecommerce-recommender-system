"""Chronological validation search helpers for recommender configurations."""

from time import perf_counter
from typing import Callable, Dict, Mapping, Sequence

import pandas as pd

from src.evaluation import evaluate_recommender
from src.recommendation import InteractionData, build_interaction_data


ModelFactory = Callable[[InteractionData, Mapping[str, object]], object]


def evaluate_recommender_grid(
    train_events: pd.DataFrame,
    validation_events: pd.DataFrame,
    user_ids: Sequence[int],
    catalog_items: Sequence[int],
    parameter_grid: Sequence[Mapping[str, object]],
    model_factory: ModelFactory,
    model_name: str,
    k: int = 10,
    relevance_mode: str = "binary",
) -> pd.DataFrame:
    """Evaluate each model configuration on one chronological validation split."""
    if not parameter_grid:
        raise ValueError("parameter_grid must contain at least one configuration.")
    if not model_name:
        raise ValueError("model_name must not be empty.")
    if k < 1:
        raise ValueError("k must be positive.")

    if "event_time" in train_events and "event_time" in validation_events:
        if not train_events.event_time.max() < validation_events.event_time.min():
            raise ValueError("Training events must precede validation events strictly.")

    interaction_data = build_interaction_data(
        train_events, user_ids=user_ids, item_ids=catalog_items
    )
    rows = []
    for parameters in parameter_grid:
        started = perf_counter()
        model = model_factory(interaction_data, parameters)
        fit_seconds = perf_counter() - started
        summary, _, _ = evaluate_recommender(
            model,
            validation_events,
            train_events,
            user_ids,
            catalog_items,
            k=k,
            relevance_mode=relevance_mode,
        )
        del model
        rows.append(
            {
                **parameters,
                **summary,
                "model": model_name,
                "fit_seconds": fit_seconds,
            }
        )
    return pd.DataFrame(rows)


def select_best_validation_configuration(
    results: pd.DataFrame,
    metric: str = "ndcg@10",
) -> pd.Series:
    """Select the first highest-scoring row, keeping ties deterministic."""
    if results.empty:
        raise ValueError("results must contain at least one evaluated configuration.")
    if metric not in results:
        raise ValueError("results must contain the '{}' metric.".format(metric))
    if results[metric].isna().any():
        raise ValueError("The selection metric must not contain missing values.")
    return results.loc[results[metric].idxmax()].copy()
