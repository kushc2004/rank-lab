from __future__ import annotations

import numpy as np
import pandas as pd

from ranklab.data.sessionize import add_ranking_contexts
from ranklab.retrieval.faiss_index import ExactInnerProductIndex
from ranklab.retrieval.two_tower import TwoTowerArtifacts


def observed_ranking_rows(
    interactions: pd.DataFrame,
    model: TwoTowerArtifacts,
    max_gap_minutes: int = 30,
    window_size: int = 20,
) -> pd.DataFrame:
    """Score only actually exposed rows for standard/randomized evaluation."""
    rows = add_ranking_contexts(interactions, max_gap_minutes, window_size)
    rows["retrieval_score"] = model.score(rows["user_id"], rows["item_id"])
    rows["retrieval_rank"] = (
        rows.groupby("context_id")["retrieval_score"]
        .rank(method="first", ascending=False)
        .astype("int32")
    )
    rows["label"] = rows["long_view"].astype("int8")
    return rows


def retrieve_candidates(
    contexts: pd.DataFrame,
    model: TwoTowerArtifacts,
    top_k: int = 200,
    index: ExactInnerProductIndex | None = None,
    popularity_weight: float | None = None,
) -> pd.DataFrame:
    """Retrieve catalog candidates for unique contexts under an explicit contract."""
    required = {"context_id", "user_id", "timestamp_ms"}
    if missing := required - set(contexts):
        raise ValueError(f"candidate contexts missing {sorted(missing)}")
    # The candidate set is created at the beginning of the synthetic context.
    # Using the final event timestamp would let earlier outcomes from that same
    # context leak into point-in-time historical features.
    unique = contexts.sort_values("timestamp_ms").drop_duplicates("context_id", keep="first")
    user_map = {int(value): index for index, value in enumerate(model.user_ids)}
    known = unique["user_id"].map(user_map).notna()
    unique = unique.loc[known].copy()
    positions = unique["user_id"].map(user_map).astype(int).to_numpy()
    weight = float((model.metadata or {}).get("popularity_weight", 0.0)) if popularity_weight is None else float(popularity_weight)
    if weight and model.item_popularity is None:
        raise ValueError("popularity-weighted retrieval requires train-only item popularity")
    if index is None:
        index = ExactInnerProductIndex.build(model.item_ids, model.item_embeddings)
    elif not np.array_equal(index.item_ids, model.item_ids):
        raise ValueError("candidate index item IDs do not match the retriever artifact")
    elif index.embeddings.shape[1] != model.item_embeddings.shape[1]:
        raise ValueError("candidate index dimension does not match the retriever artifact")
    if weight:
        # Exact score fusion is intentionally used for Pure's small catalog.
        # It keeps the hybrid comparable with the exact dense-retrieval path;
        # no approximate-index behavior is hidden in model selection.
        all_scores = model.user_embeddings[positions] @ model.item_embeddings.T
        all_scores = all_scores + weight * model.item_popularity[None, :]
        width = min(top_k, len(model.item_ids))
        top = np.argpartition(-all_scores, width - 1, axis=1)[:, :width]
        top_scores = np.take_along_axis(all_scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        positions_in_catalog = np.take_along_axis(top, order, axis=1)
        item_ids = model.item_ids[positions_in_catalog]
        scores = np.take_along_axis(all_scores, positions_in_catalog, axis=1)
    else:
        item_ids, scores = index.search(model.user_embeddings[positions], top_k)
    width = item_ids.shape[1]
    result = pd.DataFrame(
        {
            "context_id": np.repeat(unique["context_id"].to_numpy(), width),
            "user_id": np.repeat(unique["user_id"].to_numpy(), width),
            "timestamp_ms": np.repeat(unique["timestamp_ms"].to_numpy(), width),
            "item_id": item_ids.reshape(-1),
            "retrieval_score": scores.reshape(-1),
            "retrieval_rank": np.tile(np.arange(1, width + 1), len(unique)),
        }
    )
    for column in ("date", "tab", "session_position"):
        if column in unique:
            result[column] = np.repeat(unique[column].to_numpy(), width)
    return result


def _relevance_labels(exposed: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode not in {"binary", "graded"}:
        raise ValueError("relevance mode must be 'binary' or 'graded'")
    work = exposed.loc[exposed["long_view"].eq(1), [
        column for column in (
            "context_id", "item_id", "is_like", "is_follow", "is_comment",
            "is_forward", "is_profile_enter",
        ) if column in exposed
    ]].copy()
    if mode == "binary":
        work["label"] = 1
    else:
        action_columns = [
            column for column in ("is_like", "is_follow", "is_comment", "is_forward", "is_profile_enter")
            if column in work
        ]
        explicit = work[action_columns].fillna(0).gt(0).any(axis=1) if action_columns else False
        work["label"] = 1 + np.asarray(explicit, dtype=np.int8)
    return work.groupby(["context_id", "item_id"], as_index=False)["label"].max()


def attach_labels(
    candidates: pd.DataFrame,
    exposed: pd.DataFrame,
    relevance_mode: str = "binary",
) -> pd.DataFrame:
    if relevance_mode not in {"binary", "graded"}:
        raise ValueError("relevance mode must be 'binary' or 'graded'")
    binary = _relevance_labels(exposed, "binary").rename(columns={"label": "label_binary"})
    graded = _relevance_labels(exposed, "graded").rename(columns={"label": "label_graded"})
    positives = binary.merge(graded, on=["context_id", "item_id"], how="outer", validate="one_to_one")
    positives[["label_binary", "label_graded"]] = positives[
        ["label_binary", "label_graded"]
    ].fillna(0).astype("int8")
    positives["label"] = positives[f"label_{relevance_mode}"]
    result = candidates.merge(positives, on=["context_id", "item_id"], how="left")
    for column in ("label", "label_binary", "label_graded"):
        result[column] = result[column].fillna(0).astype("int8")
    # A positive absent from top-K must not silently turn the entire query into
    # an all-negative training group. Append it as a documented forced positive.
    missing = positives.merge(
        candidates[["context_id", "item_id"]].drop_duplicates(),
        on=["context_id", "item_id"], how="left", indicator=True,
    ).loc[lambda frame: frame["_merge"].eq("left_only")].drop(columns="_merge")
    if not missing.empty:
        context_columns = [
            column for column in
            ("context_id", "user_id", "timestamp_ms", "date", "tab", "session_position")
            if column in exposed
        ]
        context = exposed.sort_values("timestamp_ms").drop_duplicates(
            "context_id", keep="first"
        )[context_columns]
        missing = missing.merge(context, on="context_id", how="left", validate="many_to_one")
        missing["retrieval_score"] = float(result["retrieval_score"].min()) - 1e-6
        last_rank = result.groupby("context_id")["retrieval_rank"].max()
        missing["retrieval_rank"] = missing["context_id"].map(last_rank).fillna(0).to_numpy() + 1
        missing["forced_positive"] = 1
        result["forced_positive"] = 0
        result = pd.concat([result, missing[result.columns]], ignore_index=True)
    else:
        result["forced_positive"] = 0
    return result


def candidate_set_diagnostics(candidates: pd.DataFrame) -> dict[str, float | int | str]:
    """Describe any label-preserving candidate augmentation for ranker metrics.

    ``attach_labels`` appends positives that the retriever missed so ranker
    training groups are not silently all-negative.  Those rows are useful for
    conditional ranker learning, but they must never be read as retrieval
    wins.  Persisting this summary alongside retrieval metrics makes that
    distinction visible in every run report.
    """
    required = {"context_id", "forced_positive"}
    if missing := required - set(candidates):
        raise ValueError(f"candidate diagnostics missing {sorted(missing)}")
    forced = candidates["forced_positive"].fillna(0).astype(bool)
    total_contexts = int(candidates["context_id"].nunique())
    forced_contexts = int(candidates.loc[forced, "context_id"].nunique())
    result: dict[str, float | int | str] = {
        "candidate_rows": int(len(candidates)),
        "candidate_contexts": total_contexts,
        "forced_positive_rows": int(forced.sum()),
        "forced_positive_contexts": forced_contexts,
        "forced_positive_context_rate": float(forced_contexts / max(total_contexts, 1)),
        "metric_scope": (
            "Ranker metrics are conditional on this augmented candidate set; "
            "use outputs/metrics/retrieval.json for unaugmented full-catalog retrieval."
        ),
    }
    if "label" in candidates:
        positive = candidates["label"].fillna(0).gt(0)
        result["positive_candidate_rows"] = int(positive.sum())
        result["forced_positive_share_of_positive_rows"] = float(
            forced.sum() / max(int(positive.sum()), 1)
        )
    return result


def retrieval_metrics(
    candidates: pd.DataFrame,
    exposed: pd.DataFrame,
    k_values: tuple[int, ...] = (50, 100, 200),
) -> dict[str, float | int]:
    """Evaluate retrieval before forced-positive insertion or ranker scoring."""
    truth = (
        exposed.loc[exposed["long_view"].eq(1), ["context_id", "item_id"]]
        .drop_duplicates()
        .groupby("context_id")["item_id"].agg(set)
    )
    retrieved = candidates.sort_values(
        ["context_id", "retrieval_rank"], kind="stable"
    ).groupby("context_id", sort=False)
    records: list[dict[str, float]] = []
    for context_id, positives in truth.items():
        if context_id not in retrieved.groups:
            continue
        frame = retrieved.get_group(context_id)
        record: dict[str, float] = {}
        for k in k_values:
            items = frame.loc[frame["retrieval_rank"].le(k), "item_id"].tolist()
            hits = [position + 1 for position, item in enumerate(items) if item in positives]
            record[f"recall@{k}"] = len(set(items) & positives) / len(positives)
            record[f"mrr@{k}"] = 1.0 / min(hits) if hits else 0.0
            gains = np.asarray([1.0 if item in positives else 0.0 for item in items])
            dcg = float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))
            ideal = float(np.sum(1.0 / np.log2(np.arange(2, min(k, len(positives)) + 2))))
            record[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0
        records.append(record)
    metrics: dict[str, float | int] = {
        "positive_contexts": int(len(truth)),
        "evaluated_contexts": int(len(records)),
        "context_coverage": float(len(records) / max(len(truth), 1)),
        "catalog_items_retrieved": int(candidates["item_id"].nunique()),
    }
    if records:
        values = pd.DataFrame(records)
        metrics.update({column: float(values[column].mean()) for column in values})
    return metrics
