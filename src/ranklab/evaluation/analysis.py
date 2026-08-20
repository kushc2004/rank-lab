from __future__ import annotations

import numpy as np
import pandas as pd

from ranklab.evaluation.ranking_metrics import evaluate_ranked_impressions


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
    rng = np.random.default_rng(seed)
    array = values.to_numpy()
    boot = np.asarray([rng.choice(array, len(array), replace=True).mean() for _ in range(samples)])
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
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(differences, len(differences), replace=True).mean()
        for _ in range(samples)
    ])
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
    diversities: list[float] = []
    unique_categories: list[int] = []
    entropies: list[float] = []
    for _, frame in top.groupby("context_id", sort=False):
        categories = frame["category"].fillna("unknown").astype(str).to_numpy()
        n = len(categories)
        if n > 1:
            pairs = n * (n - 1) / 2
            same = sum(count * (count - 1) / 2 for count in pd.Series(categories).value_counts())
            diversities.append(float(1 - same / pairs))
        else:
            diversities.append(0.0)
        counts = pd.Series(categories).value_counts(normalize=True).to_numpy()
        entropy = float(-np.sum(counts * np.log(np.maximum(counts, 1e-12))))
        entropies.append(entropy / np.log(n) if n > 1 else 0.0)
        unique_categories.append(int(pd.Series(categories).nunique()))
    return {
        "intra_list_category_diversity": float(np.mean(diversities)) if diversities else float("nan"),
        "normalized_category_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "mean_unique_categories": float(np.mean(unique_categories)) if unique_categories else float("nan"),
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
