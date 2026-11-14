import pandas as pd

from src.evaluation import evaluate_recommender
from src.recommendation import (
    ImplicitLibraryALSRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RecentPopularityRecommender,
    build_interaction_data,
)


def _events(rows):
    return pd.DataFrame(rows, columns=["visitorid", "itemid", "event"])


def test_popularity_excludes_seen_items():
    train = _events(
        [
            (1, 10, "view"),
            (1, 10, "view"),
            (2, 20, "transaction"),
            (3, 30, "addtocart"),
        ]
    )
    model = PopularityRecommender().fit(train)

    assert model.recommend(1, seen_items={20}, k=2) == [30, 10]


def test_recent_popularity_uses_only_lookback_events_before_cutoff():
    events = _events(
        [
            (1, 10, "transaction"),
            (1, 20, "view"),
            (1, 30, "addtocart"),
            (1, 40, "transaction"),
        ]
    )
    events["event_time"] = pd.to_datetime(
        [
            "2019-12-31T23:59:59Z",
            "2020-01-01T00:00:00Z",
            "2020-01-14T00:00:00Z",
            "2020-01-15T00:00:00Z",
        ],
        utc=True,
    )
    model = RecentPopularityRecommender().fit(
        events,
        as_of=pd.Timestamp("2020-01-15T00:00:00Z"),
        lookback_days=14,
    )

    assert model.recommend(1, k=3) == [30, 20]


def test_item_knn_and_als_return_only_unseen_catalog_items():
    train = _events(
        [
            (1, 10, "view"),
            (1, 20, "addtocart"),
            (2, 10, "view"),
            (2, 30, "transaction"),
            (3, 20, "view"),
            (3, 30, "addtocart"),
            (4, 10, "view"),
        ]
    )
    data = build_interaction_data(train)
    for model in (
        ItemKNNRecommender(neighbors=2, shrinkage=0).fit(data),
        ImplicitLibraryALSRecommender(
            factors=2, iterations=2, random_state=1, num_threads=1
        ).fit(data),
    ):
        recommendations = model.recommend(1, seen_items={10, 20}, k=2)
        assert recommendations
        assert set(recommendations).isdisjoint({10, 20})
        assert set(recommendations).issubset({10, 20, 30})


def test_evaluation_removes_seen_targets_and_reports_catalog_coverage():
    history = _events([(1, 10, "view"), (1, 20, "addtocart")])
    target = _events([(1, 10, "transaction"), (1, 30, "transaction")])
    model = PopularityRecommender().fit(
        _events([(2, 30, "transaction"), (2, 40, "view")]),
        catalog_items=[10, 20, 30, 40],
    )

    metrics, per_user, recommendations = evaluate_recommender(
        model,
        target,
        history,
        user_ids=[1],
        catalog_items=[10, 20, 30, 40],
        k=2,
    )

    assert metrics["evaluated_users"] == 1
    assert metrics["recall@2"] == 1.0
    assert per_user.loc[0, "relevant_items"] == 1
    assert recommendations[1][0] == 30
