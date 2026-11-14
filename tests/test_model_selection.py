import pandas as pd
import pytest

from src.model_selection import (
    evaluate_recommender_grid,
    select_best_validation_configuration,
)


class _FixedRecommender:
    def __init__(self, item_id):
        self.item_id = int(item_id)

    def recommend(self, user_id, seen_items=(), k=10):
        del user_id
        return [self.item_id] if self.item_id not in set(seen_items) else []


def test_validation_grid_selects_highest_ndcg_configuration():
    train = pd.DataFrame(
        {"visitorid": [1], "itemid": [10], "event": ["view"]}
    )
    validation = pd.DataFrame(
        {"visitorid": [1], "itemid": [20], "event": ["transaction"]}
    )

    results = evaluate_recommender_grid(
        train_events=train,
        validation_events=validation,
        user_ids=[1],
        catalog_items=[10, 20, 30],
        parameter_grid=[{"item_id": 30}, {"item_id": 20}],
        model_factory=lambda data, params: _FixedRecommender(params["item_id"]),
        model_name="fixed",
        k=10,
    )

    best = select_best_validation_configuration(results)

    assert best["item_id"] == 20
    assert best["ndcg@10"] == 1.0
    assert set(results["model"]) == {"fixed"}


def test_validation_grid_rejects_empty_search():
    with pytest.raises(ValueError, match="at least one configuration"):
        evaluate_recommender_grid(
            train_events=pd.DataFrame(),
            validation_events=pd.DataFrame(),
            user_ids=[1],
            catalog_items=[10],
            parameter_grid=[],
            model_factory=lambda data, params: None,
            model_name="empty",
        )


def test_search_uses_binary_by_default_and_graded_only_explicitly():
    train = pd.DataFrame({"visitorid": [1], "itemid": [1], "event": ["view"]})
    validation = pd.DataFrame({"visitorid": [1, 1], "itemid": [10, 20],
                               "event": ["view", "transaction"]})
    kwargs = dict(train_events=train, validation_events=validation, user_ids=[1],
                  catalog_items=[1, 10, 20], parameter_grid=[{"item_id": 10}, {"item_id": 20}],
                  model_factory=lambda data, params: _FixedRecommender(params["item_id"]),
                  model_name="fixed", k=1)
    binary = evaluate_recommender_grid(**kwargs)
    graded = evaluate_recommender_grid(**kwargs, relevance_mode="graded")
    assert select_best_validation_configuration(binary, "ndcg@1")["item_id"] == 10
    assert select_best_validation_configuration(graded, "ndcg@1")["item_id"] == 20


def test_search_rejects_overlapping_time_windows():
    train = pd.DataFrame({"visitorid": [1], "itemid": [1], "event": ["view"],
                          "event_time": pd.to_datetime(["2020-01-02"], utc=True)})
    validation = train.copy()
    with pytest.raises(ValueError, match="precede validation"):
        evaluate_recommender_grid(train, validation, [1], [1], [{}],
                                  lambda data, params: _FixedRecommender(1), "fixed")
