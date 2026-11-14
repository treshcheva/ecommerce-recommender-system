"""Recommenders and bounded benchmark utilities for implicit feedback."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy import sparse
from threadpoolctl import threadpool_limits


DEFAULT_EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


@dataclass(frozen=True)
class InteractionData:
    """Sparse user-item strengths together with their row/column identifiers."""

    matrix: sparse.csr_matrix
    user_ids: List[int]
    item_ids: List[int]
    user_to_index: Dict[int, int]
    item_to_index: Dict[int, int]


def select_benchmark_users(
    train_events: pd.DataFrame,
    min_history: int = 5,
    max_users: int = 3000,
    random_state: int = 42,
) -> List[int]:
    """Choose a reproducible cohort using training history only."""
    if min_history < 1 or max_users < 1:
        raise ValueError("min_history and max_users must both be positive.")
    eligible = train_events.groupby("visitorid").size()
    eligible = eligible[eligible >= min_history].index.to_numpy()
    if len(eligible) == 0:
        raise ValueError("No users meet the minimum training-history threshold.")
    rng = np.random.RandomState(random_state)
    selected = rng.choice(eligible, size=min(max_users, len(eligible)), replace=False)
    return sorted(int(user_id) for user_id in selected)


def select_catalog_items(
    train_events: pd.DataFrame,
    max_items: int = 1500,
) -> List[int]:
    """Choose the most-interacted training items as a fixed benchmark catalog."""
    if max_items < 1:
        raise ValueError("max_items must be positive.")
    item_counts = train_events["itemid"].value_counts()
    return [int(item_id) for item_id in item_counts.head(max_items).index]


def build_interaction_data(
    events: pd.DataFrame,
    user_ids: Optional[Sequence[int]] = None,
    item_ids: Optional[Sequence[int]] = None,
    event_weights: Optional[Dict[str, float]] = None,
) -> InteractionData:
    """Build a weighted CSR matrix, summing repeated user-item events."""
    weights = DEFAULT_EVENT_WEIGHTS if event_weights is None else event_weights
    if not weights:
        raise ValueError("event_weights must contain at least one event type.")
    unknown_events = sorted(set(events["event"].unique()) - set(weights))
    if unknown_events:
        raise ValueError("No interaction weight configured for: {}".format(unknown_events))

    selected = events
    if user_ids is not None:
        selected = selected[selected["visitorid"].isin(user_ids)]
    if item_ids is not None:
        selected = selected[selected["itemid"].isin(item_ids)]

    users = (
        list(dict.fromkeys(int(value) for value in user_ids))
        if user_ids is not None
        else sorted(int(value) for value in selected["visitorid"].unique())
    )
    items = (
        list(dict.fromkeys(int(value) for value in item_ids))
        if item_ids is not None
        else sorted(int(value) for value in selected["itemid"].unique())
    )
    user_to_index = {user_id: index for index, user_id in enumerate(users)}
    item_to_index = {item_id: index for index, item_id in enumerate(items)}

    if selected.empty:
        matrix = sparse.csr_matrix((len(users), len(items)), dtype=np.float32)
    else:
        row_indices = selected["visitorid"].map(user_to_index).to_numpy()
        column_indices = selected["itemid"].map(item_to_index).to_numpy()
        values = selected["event"].map(weights).to_numpy(dtype=np.float32)
        matrix = sparse.coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(len(users), len(items)),
            dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()

    return InteractionData(
        matrix=matrix,
        user_ids=users,
        item_ids=items,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
    )


class PopularityRecommender:
    """Global event-strength popularity baseline."""

    def fit(
        self,
        events: pd.DataFrame,
        catalog_items: Optional[Sequence[int]] = None,
        event_weights: Optional[Dict[str, float]] = None,
    ) -> "PopularityRecommender":
        weights = DEFAULT_EVENT_WEIGHTS if event_weights is None else event_weights
        unknown_events = sorted(set(events["event"].unique()) - set(weights))
        if unknown_events:
            raise ValueError(
                "No interaction weight configured for: {}".format(unknown_events)
            )
        scored = events.assign(
            strength=events["event"].map(weights).astype(np.float64)
        )
        if catalog_items is not None:
            scored = scored[scored["itemid"].isin(catalog_items)]
        scores = scored.groupby("itemid")["strength"].sum().sort_values(
            ascending=False, kind="mergesort"
        )
        self.item_ids = [int(item_id) for item_id in scores.index]
        self.scores = scores.to_dict()
        return self

    def recommend(
        self,
        user_id: int,
        seen_items: Iterable[int] = (),
        k: int = 10,
    ) -> List[int]:
        del user_id
        seen: Set[int] = set(seen_items)
        if k <= 0:
            return []
        return [item_id for item_id in self.item_ids if item_id not in seen][:k]


class RecentPopularityRecommender(PopularityRecommender):
    """Popularity baseline computed from a fixed recent training window."""

    def fit(
        self,
        events: pd.DataFrame,
        as_of: pd.Timestamp,
        lookback_days: int = 14,
        catalog_items: Optional[Sequence[int]] = None,
        event_weights: Optional[Dict[str, float]] = None,
    ) -> "RecentPopularityRecommender":
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")
        boundary = pd.Timestamp(as_of)
        if boundary.tzinfo is None:
            boundary = boundary.tz_localize("UTC")
        else:
            boundary = boundary.tz_convert("UTC")
        start = boundary - pd.Timedelta(days=lookback_days)
        recent = events[
            (events["event_time"] >= start) & (events["event_time"] < boundary)
        ]
        super().fit(recent, catalog_items, event_weights)
        self.window_start = start
        self.window_end = boundary
        return self


class ItemKNNRecommender:
    """Cosine item-item collaborative filtering over a sparse interaction matrix."""

    def __init__(self, neighbors: int = 50, shrinkage: float = 10.0, block_size: int = 128):
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        self.block_size = block_size
        if neighbors < 1 or shrinkage < 0:
            raise ValueError("neighbors must be positive and shrinkage nonnegative.")
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, data: InteractionData) -> "ItemKNNRecommender":
        binary = data.matrix.copy().tocsr()
        binary.data = np.ones_like(binary.data, dtype=np.float32)
        n_items = binary.shape[1]
        item_norms = np.sqrt(np.asarray(binary.sum(axis=0)).ravel())
        # Keep the original cosine/shrinkage/top-k arithmetic and tie handling.
        # Bound intermediate co-occurrence to block_size × n_items, never I².
        item_users = binary.T.tocsr()
        capacity = n_items * min(self.neighbors, max(n_items - 1, 0))
        indices = np.empty(capacity, dtype=np.int32)
        values = np.empty(capacity, dtype=np.float32)
        indptr = np.zeros(n_items + 1, dtype=np.int32)
        offset = 0
        for start in range(0, n_items, self.block_size):
            stop = min(start + self.block_size, n_items)
            cooccurrence = (item_users[start:stop] @ binary).tocsr()
            cooccurrence.sort_indices()
            for local_index, item_index in enumerate(range(start, stop)):
                row_start, row_end = cooccurrence.indptr[local_index:local_index + 2]
                columns = cooccurrence.indices[row_start:row_end]
                counts = cooccurrence.data[row_start:row_end]
                keep = (columns != item_index) & (counts != 0)
                columns, counts = columns[keep], counts[keep]
                scores = counts / (
                    item_norms[item_index] * item_norms[columns] + self.shrinkage
                )
                if len(scores) > self.neighbors:
                    keep = np.argpartition(scores, -self.neighbors)[-self.neighbors:]
                    columns, scores = columns[keep], scores[keep]
                order = np.argsort(-scores, kind="mergesort")
                size = len(order)
                indices[offset:offset + size] = columns[order]
                values[offset:offset + size] = scores[order]
                offset += size
                indptr[item_index + 1] = offset
            del cooccurrence
        self.similarity = sparse.csr_matrix(
            (values[:offset], indices[:offset], indptr), shape=(n_items, n_items)
        )
        self.data = data
        return self

    def recommend(
        self,
        user_id: int,
        seen_items: Iterable[int] = (),
        k: int = 10,
    ) -> List[int]:
        return [
            item_id
            for item_id, _ in self.recommend_with_scores(user_id, seen_items, k)
        ]

    def recommend_with_scores(
        self,
        user_id: int,
        seen_items: Iterable[int] = (),
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        user_index = self.data.user_to_index.get(int(user_id))
        if user_index is None or k <= 0:
            return []
        profile = self.data.matrix.getrow(user_index)
        scores = (profile @ self.similarity).toarray().ravel()
        seen = set(int(item_id) for item_id in seen_items)
        for item_id in seen:
            item_index = self.data.item_to_index.get(item_id)
            if item_index is not None:
                scores[item_index] = -np.inf
        ranked_item_ids = _rank_item_ids(scores, self.data.item_ids, k)
        return [
            (item_id, float(scores[self.data.item_to_index[item_id]]))
            for item_id in ranked_item_ids
        ]


class ImplicitLibraryALSRecommender:
    """Adapter for the optimized ALS model provided by the implicit package."""

    def __init__(
        self,
        factors: int = 32,
        regularization: float = 0.1,
        iterations: int = 10,
        alpha: float = 20.0,
        random_state: int = 42,
        num_threads: int = 1,
    ):
        with threadpool_limits(limits=1, user_api="blas"):
            self.model = AlternatingLeastSquares(
                factors=factors,
                regularization=regularization,
                iterations=iterations,
                alpha=alpha,
                random_state=random_state,
                num_threads=num_threads,
            )

    def fit(self, data: InteractionData) -> "ImplicitLibraryALSRecommender":
        self.data = data
        with threadpool_limits(limits=1, user_api="blas"):
            self.model.fit(
                data.matrix.astype(np.float32).tocsr(), show_progress=False
            )
        return self

    def recommend(
        self,
        user_id: int,
        seen_items: Iterable[int] = (),
        k: int = 10,
    ) -> List[int]:
        return [
            item_id
            for item_id, _ in self.recommend_with_scores(user_id, seen_items, k)
        ]

    def recommend_with_scores(
        self,
        user_id: int,
        seen_items: Iterable[int] = (),
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        user_index = self.data.user_to_index.get(int(user_id))
        if user_index is None or k <= 0:
            return []
        seen = set(int(item_id) for item_id in seen_items)
        with threadpool_limits(limits=1, user_api="blas"):
            item_indices, scores = self.model.recommend(
                userid=user_index,
                user_items=self.data.matrix.getrow(user_index),
                N=len(self.data.item_ids),
                filter_already_liked_items=True,
            )
        return [
            (int(self.data.item_ids[item_index]), float(score))
            for item_index, score in zip(item_indices, scores)
            if int(self.data.item_ids[item_index]) not in seen
        ][:k]


def _rank_item_ids(scores: np.ndarray, item_ids: Sequence[int], k: int) -> List[int]:
    if k <= 0 or len(item_ids) == 0:
        return []
    item_array = np.asarray(item_ids)
    order = np.lexsort((item_array, -scores))
    finite_order = [index for index in order if np.isfinite(scores[index])]
    return [int(item_array[index]) for index in finite_order[:k]]
