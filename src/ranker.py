"""CatBoost learning-to-rank training and recommendation helpers."""

from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


RANKER_FEATURES = [
    "als_score",
    "als_rank",
    "has_als",
    "itemknn_score",
    "itemknn_rank",
    "has_itemknn",
    "candidate_source",
    "user_view_count",
    "user_addtocart_count",
    "user_transaction_count",
    "user_unique_items",
    "user_last_activity_hours",
    "user_total_events",
    "item_view_count",
    "item_addtocart_count",
    "item_transaction_count",
    "item_unique_users",
    "item_event_count_7d",
]
CF_FEATURES = frozenset({
    "als_score", "als_rank", "has_als",
    "itemknn_score", "itemknn_rank", "has_itemknn", "candidate_source",
})
CATEGORICAL_FEATURES = ["candidate_source"]
DEFAULT_RANKER_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "YetiRank",
    "eval_metric": "NDCG:top=10",
    "random_seed": 42,
    "verbose": 100,
}


def _prepare_grouped_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    require_target: bool,
) -> pd.DataFrame:
    required = {"snapshot_id", "user_id", "item_id"} | set(feature_columns)
    if require_target:
        required.add("target")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Ranking frame is missing columns: {}".format(sorted(missing)))
    if frame.empty:
        raise ValueError("Ranking frame must not be empty.")
    if frame.duplicated(["snapshot_id", "user_id", "item_id"]).any():
        raise ValueError("Duplicate user-item pairs within a ranking snapshot.")

    result = frame.copy()
    result["snapshot_id"] = result["snapshot_id"].astype(str)
    group_keys = sorted(
        set(zip(result["snapshot_id"], result["user_id"].astype(int)))
    )
    group_lookup = {key: index for index, key in enumerate(group_keys)}
    result["group_id"] = [
        group_lookup[(snapshot_id, int(user_id))]
        for snapshot_id, user_id in zip(result["snapshot_id"], result["user_id"])
    ]
    result = result.sort_values(
        ["group_id", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    group_order = result["group_id"].to_numpy()
    group_starts = np.r_[True, group_order[1:] != group_order[:-1]]
    if len(group_order[group_starts]) != int(result["group_id"].nunique()):
        raise RuntimeError("Rows for a ranking group are not contiguous.")
    return result


def make_ranker_pool(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] = RANKER_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> Pool:
    """Create a CatBoost Pool with contiguous, unique user-snapshot groups."""
    grouped = _prepare_grouped_frame(frame, feature_columns, require_target=True)
    if not set(categorical_features).issubset(feature_columns):
        raise ValueError("Categorical features must be included in feature_columns.")
    if not grouped["target"].isin([0, 1]).all():
        raise ValueError("The initial ranker target must be binary (0 or 1).")
    return Pool(
        data=grouped[list(feature_columns)],
        label=grouped["target"].astype(int),
        group_id=grouped["group_id"].astype(int),
        cat_features=list(categorical_features),
        feature_names=list(feature_columns),
    )


def fit_catboost_ranker(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    params: Optional[Dict[str, object]] = None,
    early_stopping_rounds: int = 50,
    feature_columns: Sequence[str] = RANKER_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> CatBoostRanker:
    """Fit CatBoost on training snapshots and select iteration on validation."""
    train = train_frame[
        train_frame.groupby(["snapshot_id", "user_id"])["target"].transform("max") > 0
    ]
    validation = validation_frame[
        validation_frame.groupby(["snapshot_id", "user_id"])["target"].transform("max")
        > 0
    ]
    if train.empty:
        raise ValueError("No training user groups contain a positive candidate.")
    if validation.empty:
        raise ValueError("No validation user groups contain a positive candidate.")
    model_params = dict(DEFAULT_RANKER_PARAMS)
    if params is not None:
        model_params.update(params)
    model = CatBoostRanker(**model_params)
    train_pool = make_ranker_pool(
        train,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
    )
    validation_pool = make_ranker_pool(
        validation,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
    )
    model.fit(
        train_pool,
        eval_set=validation_pool,
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True,
    )
    return model


class CandidateRankerRecommender:
    """Bind recommendations to one snapshot; never pool past and future rows."""

    def __init__(
        self,
        model: CatBoostRanker,
        candidate_features: pd.DataFrame,
        feature_columns: Sequence[str] = RANKER_FEATURES,
        snapshot_id: Optional[str] = None,
    ):
        if "snapshot_id" not in candidate_features:
            raise ValueError("candidate_features must contain snapshot_id.")
        if candidate_features["snapshot_id"].isna().any():
            raise ValueError("snapshot_id must not be missing.")
        snapshots = candidate_features["snapshot_id"].astype(str)
        if snapshot_id is None:
            if snapshots.nunique() != 1:
                raise ValueError("Specify snapshot_id when candidates contain multiple snapshots.")
            snapshot_id = snapshots.iloc[0]
        self.snapshot_id = str(snapshot_id)
        selected = candidate_features.loc[snapshots == self.snapshot_id].copy()
        if selected.empty:
            raise ValueError("No candidates for snapshot_id: {}".format(snapshot_id))
        grouped = _prepare_grouped_frame(
            selected, feature_columns, require_target=False
        )
        self.candidates = grouped.copy()
        self.candidates["ranker_score"] = model.predict(
            self.candidates[list(feature_columns)]
        )
        self.candidates = self.candidates.sort_values(
            ["user_id", "ranker_score", "item_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )

    def recommend(
        self,
        user_id: int,
        seen_items: Iterable[int] = (),
        k: int = 10,
    ) -> List[int]:
        if k <= 0:
            return []
        seen: Set[int] = set(int(item_id) for item_id in seen_items)
        user_candidates = self.candidates[
            (self.candidates["user_id"] == int(user_id))
            & (~self.candidates["item_id"].isin(seen))
        ]
        return [int(item_id) for item_id in user_candidates["item_id"].head(k)]


def feature_importance_frame(
    model: CatBoostRanker,
    data: Pool,
    feature_columns: Sequence[str] = RANKER_FEATURES,
) -> pd.DataFrame:
    """Return CatBoost feature importances in descending order."""
    importance = model.get_feature_importance(
        data=data, type="LossFunctionChange"
    )
    if len(importance) != len(feature_columns):
        raise ValueError("Feature importance output does not match feature_columns.")
    return (
        pd.DataFrame({"feature": list(feature_columns), "importance": importance})
        .sort_values("importance", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
