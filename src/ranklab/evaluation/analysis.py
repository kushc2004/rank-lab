from __future__ import annotations

import numpy as np
import pandas as pd

from ranklab.evaluation.ranking_metrics import evaluate_ranked_impressions


def _bootstrap_means(values: np.ndarray, samples: int, seed: int) -> np.ndarray:
    """Compute bootstrap means in vectorized, bounded-memory batches.

    The previous implementation made one Python/NumPy call per replicate.  On
    the full KuaiRand evaluation this dominated the analysis wall-clock time.
    Batching keeps the identical user-level resampling protocol, seed, and
    number of replicates while bounding the temporary draw matrix.
    """
    array = np.asarray(values, dtype=float)
    if samples < 1:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(seed)
    # At most roughly 16 MiB of float64 sampled values per batch.  The index
    # matrix used internally by NumPy may add memory, so keep this conservative.
    batch_size = max(1, min(samples, 2_000_000 // max(len(array), 1)))
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        current = min(batch_size, remaining)
        draws = rng.choice(array, size=(current, len(array)), replace=True)
        means.append(draws.mean(axis=1))
        remaining -= current
    return np.concatenate(means)


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not len(values) or np.allclose(values.sum(), 0):
        return 0.0
    values = np.sort(np.maximum(values, 0))
    n = len(values)
    return float((2 * np.arange(1, n + 1) @ values) / (n * values.sum()) - (n + 1) / n)


def recommendation_behavior(predictions: pd.DataFrame, catalog_size: int, k: int = 10) -> dict[str, float]:
    top = predictions.sort_values(
        ["context_id", "score"], ascending=[True, False], kind="stable"
    ).groupby("context_id", sort=False).head(k)
    frequency = top["item_id"].value_counts()
    return {
        "catalog_coverage": float(frequency.size / max(catalog_size, 1)),
        "recommendation_gini": gini(frequency.to_numpy()),
        "unique_recommended_items": int(frequency.size),
    }


def bootstrap_user_metric(
    per_group: pd.DataFrame,
    metric: str = "ndcg",
    k: int = 10,
    samples: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    values = per_group.loc[per_group["k"].eq(k)].groupby("user_id")[metric].mean()
    if values.empty:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_users": 0}
    array = values.to_numpy()
    boot = _bootstrap_means(array, samples, seed)
    return {
        "mean": float(array.mean()), "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)), "n_users": int(len(array)),
    }


def paired_user_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str = "ndcg",
    k: int = 10,
    samples: int = 500,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired user-level confidence interval for left minus right."""
    left_user = left.loc[left["k"].eq(k)].groupby("user_id")[metric].mean().rename("left")
    right_user = right.loc[right["k"].eq(k)].groupby("user_id")[metric].mean().rename("right")
    paired = pd.concat([left_user, right_user], axis=1, join="inner").dropna()
    if paired.empty:
        return {"mean_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_users": 0}
    differences = (paired["left"] - paired["right"]).to_numpy()
    boot = _bootstrap_means(differences, samples, seed)
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "n_users": int(len(differences)),
    }


def cohort_report(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    users = predictions[["user_id"]].drop_duplicates().merge(
        train.groupby("user_id").size().rename("history_count"),
        on="user_id", how="left",
    )
    users["history_count"] = users["history_count"].fillna(0)
    users["cohort"] = pd.qcut(
        users["history_count"].rank(method="first"),
        3,
        labels=["low", "medium", "high"],
    )
    work = predictions.merge(users, on="user_id", how="left", validate="many_to_one")
    records = []
    for cohort, frame in work.groupby("cohort", observed=True):
        metrics, _ = evaluate_ranked_impressions(frame, k_values=k_values)
        records.append({"cohort": str(cohort), "users": frame.user_id.nunique(), "rows": len(frame), **metrics})
    return pd.DataFrame(records)


def item_popularity_cohorts(train: pd.DataFrame) -> pd.DataFrame:
    exposure = train.groupby("item_id").size().rename("training_exposure").reset_index()
    ordered = exposure.sort_values(["training_exposure", "item_id"], ascending=[False, True]).reset_index(drop=True)
    percentile = (ordered.index.to_numpy() + 1) / max(len(ordered), 1)
    ordered["item_cohort"] = np.select(
        [percentile <= 0.10, percentile <= 0.50], ["head", "mid"], default="tail"
    )
    return ordered


def item_cohort_report(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Popularity/freshness analysis based only on pre-evaluation exposure."""
    cohorts = item_popularity_cohorts(train)
    work = predictions.merge(cohorts, on="item_id", how="left", validate="many_to_one")
    work["training_exposure"] = work["training_exposure"].fillna(0)
    work["item_cohort"] = work["item_cohort"].fillna("zero_history")
    ranked = work.sort_values(["context_id", "score"], ascending=[True, False], kind="stable")
    ranked["rank"] = ranked.groupby("context_id", sort=False).cumcount() + 1
    records: list[dict[str, float | int | str]] = []
    total_interactions = max(len(work), 1)
    for cohort, frame in work.groupby("item_cohort", observed=True):
        top = ranked.loc[ranked["item_cohort"].eq(cohort) & ranked["rank"].le(max(k_values))]
        record: dict[str, float | int | str] = {
            "cohort": str(cohort),
            "catalog_items": int(frame["item_id"].nunique()),
            "rows": int(len(frame)),
            "interaction_share": float(len(frame) / total_interactions),
            "positive_rate": float(frame["long_view"].mean()),
            "recommended_share": float(len(top) / max(len(ranked.loc[ranked["rank"].le(max(k_values))]), 1)),
            "average_training_exposure": float(top["training_exposure"].mean()) if not top.empty else 0.0,
        }
        for k in k_values:
            at_k = ranked.loc[ranked["rank"].le(k)]
            record[f"recommendation_share@{k}"] = float(at_k["item_cohort"].eq(cohort).mean())
            positives = work.loc[work["item_cohort"].eq(cohort) & work["long_view"].eq(1), ["context_id", "item_id"]].drop_duplicates()
            hits = at_k.loc[at_k["item_cohort"].eq(cohort) & at_k["long_view"].eq(1), ["context_id", "item_id"]].drop_duplicates()
            record[f"positive_recall@{k}"] = float(len(hits) / max(len(positives), 1))
        records.append(record)
    return pd.DataFrame(records)


def diversity_report(predictions: pd.DataFrame, k: int = 10) -> dict[str, float | int]:
    top = predictions.sort_values(
        ["context_id", "score"], ascending=[True, False], kind="stable"
    ).groupby("context_id", sort=False).head(k)
    if top.empty:
        return {
            "intra_list_category_diversity": float("nan"),
            "normalized_category_entropy": float("nan"),
            "mean_unique_categories": float("nan"),
            "contexts": 0,
            "k": int(k),
        }
    work = top[["context_id", "category"]].copy()
    work["category"] = work["category"].fillna("unknown").astype(str)
    counts = work.groupby(["context_id", "category"], sort=False).size().rename("count")
    group_counts = counts.groupby(level="context_id", sort=False)
    sizes = group_counts.sum().astype(float)
    same_pairs = (counts * (counts - 1) / 2).groupby(level="context_id", sort=False).sum()
    total_pairs = sizes * (sizes - 1) / 2
    diversity = (1 - same_pairs / total_pairs).where(sizes.gt(1), 0.0)
    probabilities = counts / group_counts.transform("sum")
    entropy_terms = -(probabilities * np.log(np.maximum(probabilities, 1e-12)))
    entropy = entropy_terms.groupby(level="context_id", sort=False).sum()
    normalized_entropy = (entropy / np.log(sizes)).where(sizes.gt(1), 0.0)
    unique_categories = group_counts.size()
    return {
        "intra_list_category_diversity": float(diversity.mean()),
        "normalized_category_entropy": float(normalized_entropy.mean()),
        "mean_unique_categories": float(unique_categories.mean()),
        "contexts": int(len(unique_categories)),
        "k": int(k),
    }


def model_ranking_stability(metrics: dict[str, dict], metric: str = "ndcg@10") -> dict[str, object]:
    rows = []
    for model, values in metrics.items():
        if "standard" in values and "randomized" in values and metric in values["standard"] and metric in values["randomized"]:
            rows.append({"model": model, "standard": values["standard"][metric], "randomized": values["randomized"][metric]})
    frame = pd.DataFrame(rows)
    correlation = float(frame["standard"].rank().corr(frame["randomized"].rank(), method="pearson")) if len(frame) > 1 else float("nan")
    return {
        "metric": metric,
        "spearman": correlation,
        "model_count": int(len(frame)),
        "standard_order": frame.sort_values("standard", ascending=False)["model"].tolist() if not frame.empty else [],
        "randomized_order": frame.sort_values("randomized", ascending=False)["model"].tolist() if not frame.empty else [],
        "interpretation": "Cautious: rank correlation is unstable with few models.",
    }
