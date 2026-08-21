import numpy as np
import pandas as pd

from ranklab.ranking.candidate_builder import retrieve_candidates
from ranklab.retrieval.two_tower import TwoTowerArtifacts


def test_popularity_hybrid_changes_catalog_order_only_when_selected():
    model = TwoTowerArtifacts(
        user_ids=np.asarray([1]),
        item_ids=np.asarray([10, 11]),
        user_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        item_embeddings=np.asarray([[1.0, 0.0], [0.8, 0.0]], dtype=np.float32),
        device="cpu",
        metadata={"popularity_weight": 0.0},
        item_popularity=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    contexts = pd.DataFrame({"context_id": ["c"], "user_id": [1], "timestamp_ms": [1]})

    pure = retrieve_candidates(contexts, model, top_k=2)
    hybrid = retrieve_candidates(contexts, model, top_k=2, popularity_weight=0.5)

    assert pure.iloc[0]["item_id"] == 10
    assert hybrid.iloc[0]["item_id"] == 11


def test_artifact_score_uses_persisted_selected_hybrid_weight():
    model = TwoTowerArtifacts(
        user_ids=np.asarray([1]), item_ids=np.asarray([10, 11]),
        user_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        item_embeddings=np.asarray([[1.0, 0.0], [0.8, 0.0]], dtype=np.float32),
        device="cpu", metadata={"popularity_weight": 0.5},
        item_popularity=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    scores = model.score(pd.Series([1, 1]), pd.Series([10, 11]))
    assert scores[1] > scores[0]
