import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.recommendation import ItemKNNRecommender, build_interaction_data


def legacy_similarity(data, neighbors, shrinkage):
    """Original full co-occurrence/LIL computation, for equivalence testing."""
    binary = data.matrix.copy()
    binary.data[:] = 1
    cooccurrence = (binary.T @ binary).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    norms = np.sqrt(np.asarray(binary.sum(axis=0)).ravel())
    similarity = cooccurrence.tolil(copy=True)
    for index in range(similarity.shape[0]):
        columns = np.asarray(similarity.rows[index], dtype=np.intp)
        counts = np.asarray(similarity.data[index], dtype=np.float32)
        scores = counts / (norms[index] * norms[columns] + shrinkage)
        if len(scores) > neighbors:
            keep = np.argpartition(scores, -neighbors)[-neighbors:]
            columns, scores = columns[keep], scores[keep]
        order = np.argsort(-scores, kind="mergesort")
        similarity.rows[index] = columns[order].tolist()
        similarity.data[index] = scores[order].tolist()
    return similarity.tocsr()


@pytest.mark.parametrize("block_size", [1, 4, 128])
@pytest.mark.parametrize("neighbors,shrinkage", [(2, 0.), (4, 10.), (100, 0.)])
def test_block_knn_preserves_original_scores_ties_and_recommendations(block_size, neighbors, shrinkage):
    rng = np.random.RandomState(42)
    events = pd.DataFrame({"visitorid": rng.randint(0, 20, 200),
                           "itemid": rng.randint(0, 17, 200), "event": "view"})
    data = build_interaction_data(events, user_ids=list(range(22)), item_ids=list(range(19)))
    expected = legacy_similarity(data, neighbors, shrinkage)
    actual = ItemKNNRecommender(neighbors, shrinkage, block_size).fit(data)
    assert (actual.similarity != expected).nnz == 0
    assert actual.similarity.nnz <= len(data.item_ids) * neighbors
    reference = ItemKNNRecommender(neighbors, shrinkage)
    reference.data, reference.similarity = data, expected
    for user in data.user_ids:
        seen = set(events.loc[events.visitorid == user, "itemid"])
        assert actual.recommend_with_scores(user, seen, k=10) == reference.recommend_with_scores(user, seen, k=10)


def test_block_knn_empty_catalog_and_invalid_block():
    data = build_interaction_data(pd.DataFrame(columns=["visitorid", "itemid", "event"]), user_ids=[1], item_ids=[])
    assert ItemKNNRecommender().fit(data).similarity.shape == (0, 0)
    with pytest.raises(ValueError, match="block_size"):
        ItemKNNRecommender(block_size=0)
