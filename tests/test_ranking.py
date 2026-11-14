import numpy as np
import pandas as pd
import pytest

from catboost import Pool

from src.candidate_generation import (
    candidate_recall,
    generate_candidate_frame,
)
from src.evaluation import evaluate_recommender
from src.ranker import (
    CandidateRankerRecommender,
    CF_FEATURES,
    CATEGORICAL_FEATURES,
    RANKER_FEATURES,
    _prepare_grouped_frame,
    fit_catboost_ranker,
    make_ranker_pool,
)
from src.ranking_features import (
    attach_binary_targets,
    build_ranking_features,
    split_history_target,
)


def _events(rows):
    return pd.DataFrame(
        rows, columns=["visitorid", "itemid", "event", "event_time"]
    ).assign(event_time=lambda frame: pd.to_datetime(frame["event_time"], utc=True))


class _ScoredModel:
    def __init__(self, ranked):
        self.ranked = ranked

    def recommend_with_scores(self, user_id, seen_items=(), k=10):
        seen = set(seen_items)
        return [
            pair for pair in self.ranked.get(user_id, [])
            if pair[0] not in seen
        ][:k]


def _candidate_frame():
    return pd.DataFrame(
        [
            {
                "user_id": 1,
                "item_id": 10,
                "als_score": 0.9,
                "als_rank": 1,
                "has_als": True,
                "itemknn_score": np.nan,
                "itemknn_rank": np.nan,
                "has_itemknn": False,
                "candidate_source": "als_only",
            },
            {
                "user_id": 1,
                "item_id": 20,
                "als_score": np.nan,
                "als_rank": np.nan,
                "has_als": False,
                "itemknn_score": 0.8,
                "itemknn_rank": 1,
                "has_itemknn": True,
                "candidate_source": "itemknn_only",
            },
        ]
    )


def test_candidate_union_keeps_source_scores_and_excludes_seen_items():
    history = pd.DataFrame(
        {"visitorid": [1], "itemid": [99]}
    )
    als = _ScoredModel({1: [(10, 0.9), (99, 0.8)]})
    itemknn = _ScoredModel({1: [(10, 0.7), (20, 0.6)]})

    candidates = generate_candidate_frame([1], history, als, itemknn, 100)

    assert set(candidates["item_id"]) == {10, 20}
    both = candidates.set_index("item_id").loc[10]
    assert both["candidate_source"] == "both"
    assert both["als_score"] == 0.9
    assert both["itemknn_score"] == 0.7
    itemknn_only = candidates.set_index("item_id").loc[20]
    assert pd.isna(itemknn_only["als_score"])
    assert not itemknn_only["has_als"]


def test_temporal_snapshot_uses_distinct_timestamps():
    events = _events(
        [
            (1, 10, "view", "2020-01-01T00:00:00Z"),
            (2, 20, "view", "2020-01-02T00:00:00Z"),
            (1, 30, "view", "2020-01-03T00:00:00Z"),
            (2, 40, "transaction", "2020-01-04T00:00:00Z"),
        ]
    )

    history, target, boundary = split_history_target(events, 0.5)

    assert history["event_time"].max() < target["event_time"].min()
    assert target["event_time"].min() == boundary


def test_feature_builder_rejects_events_at_or_after_snapshot():
    history = _events(
        [
            (1, 10, "view", "2020-01-01T00:00:00Z"),
            (1, 20, "addtocart", "2020-01-02T00:00:00Z"),
        ]
    )
    future_history = pd.concat(
        [
            history,
            _events([(1, 30, "transaction", "2020-01-03T00:00:00Z")]),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="at or after the snapshot"):
        build_ranking_features(
            future_history,
            _candidate_frame(),
            pd.Timestamp("2020-01-03T00:00:00Z"),
            "validation",
        )


def test_features_and_targets_are_built_from_distinct_time_windows():
    history = _events(
        [
            (1, 10, "view", "2020-01-01T00:00:00Z"),
            (1, 20, "addtocart", "2020-01-02T00:00:00Z"),
        ]
    )
    target = _events(
        [
            (1, 10, "transaction", "2020-01-03T00:00:00Z"),
            (1, 30, "view", "2020-01-04T00:00:00Z"),
        ]
    )
    cutoff = pd.Timestamp("2020-01-03T00:00:00Z")
    features = build_ranking_features(history, _candidate_frame(), cutoff, "train")
    labeled = attach_binary_targets(features, target, cutoff)

    assert labeled.set_index("item_id")["target"].to_dict() == {10: 1, 20: 0}
    assert labeled.loc[labeled["item_id"] == 10, "user_transaction_count"].item() == 0
    with pytest.raises(ValueError, match="before the snapshot"):
        attach_binary_targets(features, history, cutoff)


def test_ranker_pool_has_unique_contiguous_user_groups():
    frame = _candidate_frame()
    frame["snapshot_id"] = "train"
    frame["target"] = [1, 0]
    frame = pd.concat(
        [
            frame,
            frame.assign(user_id=2, item_id=lambda data: data["item_id"] + 100),
        ],
        ignore_index=True,
    )

    feature_columns = ["als_score", "candidate_source"]
    grouped = _prepare_grouped_frame(frame, feature_columns, require_target=True)
    pool = make_ranker_pool(frame, feature_columns=feature_columns)

    assert isinstance(pool, Pool)
    assert grouped["group_id"].tolist() == [0, 0, 1, 1]
    assert pool.num_row() == 4
    assert set(CATEGORICAL_FEATURES).issubset(RANKER_FEATURES)
    with pytest.raises(ValueError, match="Duplicate user-item"):
        make_ranker_pool(
            pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            feature_columns=feature_columns,
        )


def test_catboost_ranker_training_is_reproducible_with_fixed_seed():
    def examples(snapshot_id):
        rows = []
        for user_id in range(1, 5):
            for item_offset in range(4):
                row = {
                    "snapshot_id": snapshot_id,
                    "user_id": user_id,
                    "item_id": 100 + item_offset,
                    "target": int(item_offset == user_id % 4),
                    "candidate_source": "both" if item_offset % 2 == 0 else "als_only",
                }
                for feature in RANKER_FEATURES:
                    row.setdefault(feature, float(user_id + item_offset))
                row["has_als"] = True
                row["has_itemknn"] = item_offset % 2 == 0
                rows.append(row)
        return pd.DataFrame(rows)

    train = examples("train")
    validation = examples("validation")
    params = {
        "iterations": 8,
        "depth": 2,
        "learning_rate": 0.1,
        "random_seed": 17,
        "thread_count": 1,
        "verbose": False,
        "allow_writing_files": False,
    }

    selected_features = ["als_score", "candidate_source"]
    first = fit_catboost_ranker(
        train,
        validation,
        params,
        early_stopping_rounds=3,
        feature_columns=selected_features,
        categorical_features=["candidate_source"],
    )
    second = fit_catboost_ranker(
        train,
        validation,
        params,
        early_stopping_rounds=3,
        feature_columns=selected_features,
        categorical_features=["candidate_source"],
    )

    np.testing.assert_allclose(
        first.predict(train[selected_features]),
        second.predict(train[selected_features]),
    )


def test_candidate_recall_reports_both_sources_and_union():
    candidates = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 20],
            "als_rank": [1, np.nan],
            "itemknn_rank": [np.nan, 1],
        }
    )
    target = pd.DataFrame({"visitorid": [1], "itemid": [10]})
    history = pd.DataFrame({"visitorid": [1], "itemid": [99]})

    recall = candidate_recall(
        candidates, target, history, [1], [10, 20], k=100
    ).set_index("generator")

    assert recall.loc["implicit ALS", "recall"] == 1.0
    assert recall.loc["ItemKNN", "recall"] == 0.0
    assert recall.iloc[2]["recall"] == 1.0
    assert recall.iloc[2]["metric"] == "Recall@200"
    assert recall.iloc[2]["cutoff"] == 200
    assert recall.loc["implicit ALS", "metric"] == "Recall@100"


def test_binary_evaluation_treats_all_target_event_types_equally():
    class FixedRecommender:
        def recommend(self, user_id, seen_items=(), k=10):
            return [10, 20][:k]

    target = pd.DataFrame(
        {
            "visitorid": [1, 1],
            "itemid": [10, 20],
            "event": ["view", "transaction"],
        }
    )
    history = pd.DataFrame(columns=["visitorid", "itemid", "event"])
    _, binary, _ = evaluate_recommender(
        FixedRecommender(), target, history, [1], [10, 20], k=2,
        relevance_mode="binary",
    )
    _, graded, _ = evaluate_recommender(
        FixedRecommender(), target, history, [1], [10, 20], k=2,
        relevance_mode="graded",
    )

    assert binary.loc[0, "ndcg@2"] == 1.0
    assert graded.loc[0, "ndcg@2"] < 1.0


def test_ranker_requires_snapshot_and_never_predicts_other_periods():
    class ScoreModel:
        def predict(self, features):
            # A future candidate must not even be sent to the model.
            assert features["als_score"].max() < 10
            return features["als_score"].to_numpy()

    past = pd.DataFrame({"snapshot_id": ["past", "past"], "user_id": [1, 1],
                         "item_id": [10, 20], "als_score": [1., 2.]})
    future = past.assign(snapshot_id="future", als_score=100.)
    combined = pd.concat([past, future], ignore_index=True)
    with pytest.raises(ValueError, match="Specify snapshot_id"):
        CandidateRankerRecommender(ScoreModel(), combined, ["als_score"])
    model = CandidateRankerRecommender(ScoreModel(), combined, ["als_score"], snapshot_id="past")
    assert model.recommend(1) == [20, 10]
    assert model.recommend(1, seen_items=[20]) == [10]
    assert model.recommend(2) == []
    assert model.recommend(1, k=0) == []
    assert CandidateRankerRecommender(ScoreModel(), past, ["als_score"]).recommend(1) == [20, 10]
    with pytest.raises(ValueError, match="No candidates"):
        CandidateRankerRecommender(ScoreModel(), combined, ["als_score"], snapshot_id="missing")


def test_candidate_union_respects_requested_source_cutoff_and_missing_users():
    candidates = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [10, 20, 30],
                               "als_rank": [1, 2, np.nan], "itemknn_rank": [np.nan, np.nan, 1]})
    targets = pd.DataFrame({"visitorid": [1, 1, 2], "itemid": [20, 30, 30]})
    history = pd.DataFrame(columns=["visitorid", "itemid"])
    result = candidate_recall(candidates, targets, history, [1, 2], [10, 20, 30], k=1)
    union = result.iloc[2]
    assert union["cutoff"] == 2
    assert union["metric"] == "Recall@2"
    # Item 20 is outside top-1 of both sources; user 2 has no candidates.
    assert union["recall"] == 0.25
    assert union["users"] == 2


def test_default_binary_matches_explicit_binary_and_preserves_hit_metrics():
    class Fixed:
        def recommend(self, user_id, seen_items=(), k=10):
            return [10, 30][:k]
    targets = pd.DataFrame({"visitorid": [1, 1, 1], "itemid": [10, 20, 20],
                            "event": ["view", "transaction", "transaction"]})
    history = pd.DataFrame(columns=["visitorid", "itemid", "event"])
    default, _, _ = evaluate_recommender(Fixed(), targets, history, [1], [10, 20, 30], k=2)
    binary, _, _ = evaluate_recommender(Fixed(), targets, history, [1], [10, 20, 30], k=2, relevance_mode="binary")
    graded, _, _ = evaluate_recommender(Fixed(), targets, history, [1], [10, 20, 30], k=2, relevance_mode="graded")
    assert default == binary
    assert binary["ndcg@2"] > graded["ndcg@2"]
    for metric in ["recall@2", "precision@2"]:
        assert binary[metric] == graded[metric] == 0.5


def test_full_cf_ablation_retains_only_aggregates_and_can_fit_without_categories():
    features = [feature for feature in RANKER_FEATURES if feature not in CF_FEATURES]
    assert len(features) == 11
    assert all(feature.startswith(("user_", "item_")) for feature in features)
    frame = pd.DataFrame({"snapshot_id": ["train"] * 4, "user_id": [1, 1, 2, 2],
                          "item_id": [10, 20, 10, 20], "target": [1, 0, 0, 1]})
    for index, feature in enumerate(features):
        frame[feature] = np.arange(4) + index
    model = fit_catboost_ranker(frame, frame.assign(snapshot_id="validation"),
                               params={"iterations": 3, "depth": 2, "thread_count": 1,
                                       "verbose": False, "allow_writing_files": False},
                               feature_columns=features, categorical_features=[])
    assert model.feature_names_ == features
