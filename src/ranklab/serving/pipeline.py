from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import pandas as pd

from ranklab.retrieval.faiss_index import ExactInnerProductIndex
from ranklab.retrieval.two_tower import TwoTowerArtifacts
from ranklab.ranking.ranker_features import FEATURE_COLUMNS, training_aggregates


@dataclass
class RetrievalRankingPipeline:
    retrieval: TwoTowerArtifacts
    ranker: object | None = None
    training_history: pd.DataFrame | None = None
    index: ExactInnerProductIndex | None = None

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = ExactInnerProductIndex.build(
                self.retrieval.item_ids, self.retrieval.item_embeddings
            )
        if not np.array_equal(self.index.item_ids, self.retrieval.item_ids):
            raise ValueError("serving index item IDs disagree with retrieval artifacts")
        self.user_positions = {int(value): position for position, value in enumerate(self.retrieval.user_ids)}
        if self.training_history is not None:
            self.item_history, self.user_history = training_aggregates(self.training_history)
        else:
            self.item_history, self.user_history = None, None

    def recommend(self, user_id: int, timestamp_ms: int, candidate_k: int = 200, final_k: int = 10) -> pd.DataFrame:
        result, _ = self._recommend_with_timings(user_id, timestamp_ms, candidate_k, final_k)
        return result

    def _recommend_with_timings(
        self,
        user_id: int,
        timestamp_ms: int,
        candidate_k: int = 200,
        final_k: int = 10,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        total_started = time.perf_counter()
        encode_started = time.perf_counter()
        position = self.user_positions.get(int(user_id))
        if position is None:
            raise KeyError(f"unknown user_id {user_id}")
        user_embedding = self.retrieval.user_embeddings[position:position + 1]
        encode_ms = (time.perf_counter() - encode_started) * 1000

        retrieval_started = time.perf_counter()
        item_ids, scores = self.index.search(user_embedding, candidate_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        feature_started = time.perf_counter()
        result = pd.DataFrame({"user_id": user_id, "item_id": item_ids[0], "timestamp_ms": timestamp_ms, "retrieval_score": scores[0]})
        result["retrieval_rank"] = np.arange(1, len(result) + 1)
        if self.ranker is not None:
            if self.item_history is None or self.user_history is None:
                raise ValueError("a ranker requires leakage-safe training_history")
            result = result.merge(self.item_history, on="item_id", how="left", validate="many_to_one")
            result = result.merge(self.user_history, on="user_id", how="left", validate="many_to_one")
            timestamp = pd.to_datetime(result["timestamp_ms"], unit="ms", utc=True)
            result["hour"] = timestamp.dt.hour
            result["day_of_week"] = timestamp.dt.dayofweek
            result["tab"] = 0
            result["session_position"] = 0
            for column in FEATURE_COLUMNS:
                if column not in result:
                    result[column] = 0.0
                result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0)
            feature_ms = (time.perf_counter() - feature_started) * 1000
            rank_started = time.perf_counter()
            result["score"] = self.ranker.predict(result)
            ranking_ms = (time.perf_counter() - rank_started) * 1000
        else:
            feature_ms = (time.perf_counter() - feature_started) * 1000
            rank_started = time.perf_counter()
            result["score"] = result["retrieval_score"]
            ranking_ms = (time.perf_counter() - rank_started) * 1000
        result = result.sort_values(["score", "item_id"], ascending=[False, True]).head(final_k).reset_index(drop=True)
        timings = {
            "user_encoding_ms": encode_ms,
            "retrieval_ms": retrieval_ms,
            "feature_join_ms": feature_ms,
            "ranking_ms": ranking_ms,
            "total_ms": (time.perf_counter() - total_started) * 1000,
        }
        return result, timings

    def benchmark(self, user_ids: list[int], timestamp_ms: int, candidate_k: int = 200) -> dict[str, object]:
        latency: dict[str, list[float]] = {
            "user_encoding_ms": [], "retrieval_ms": [], "feature_join_ms": [],
            "ranking_ms": [], "total_ms": [],
        }
        for user_id in user_ids:
            _, timings = self._recommend_with_timings(user_id, timestamp_ms, candidate_k)
            for name, value in timings.items():
                latency[name].append(value)
        stages = {
            name: {
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "mean_ms": float(np.mean(values)),
            }
            for name, values in latency.items() if values
        }
        total = stages.get("total_ms", {})
        return {
            "requests": len(user_ids),
            "p50_ms": total.get("p50_ms", float("nan")),
            "p95_ms": total.get("p95_ms", float("nan")),
            "stages": stages,
            "label": "local benchmark; not a production SLA",
        }
