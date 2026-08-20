from __future__ import annotations

import json
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(figure: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def _placeholder(path: Path, title: str, reason: str) -> str:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title, pad=20)
    axis.text(
        0.5, 0.5, reason, ha="center", va="center", wrap=True,
        transform=axis.transAxes, color="#555555",
    )
    return _save(figure, path)


def _standard_random_exposure(interactions: pd.DataFrame, path: Path) -> str:
    counts = (
        interactions.assign(
            evaluation_policy=np.where(
                interactions["policy"].eq("random"), "randomized", "standard"
            )
        )
        .groupby(["evaluation_policy", "item_id"], observed=True)
        .size()
        .rename("exposures")
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for policy, frame in counts.groupby("evaluation_policy", sort=True):
        ordered = np.sort(frame["exposures"].to_numpy(dtype=float))[::-1]
        axis.plot(np.arange(1, len(ordered) + 1), ordered, label=str(policy))
    axis.set(xlabel="Item exposure rank", ylabel="Exposure count", yscale="log")
    axis.set_title("Standard vs randomized item exposure")
    axis.legend()
    return _save(figure, path)


def _standard_random_rewards(interactions: pd.DataFrame, path: Path) -> str:
    reward_columns = [
        value for value in ("is_click", "long_view", "is_like", "is_follow")
        if value in interactions
    ]
    if not reward_columns:
        return _placeholder(path, "Standard vs randomized reward rates", "No reward columns are available.")
    work = interactions.copy()
    work["evaluation_policy"] = np.where(
        work["policy"].eq("random"), "randomized", "standard"
    )
    values = work.groupby("evaluation_policy", observed=True)[reward_columns].mean().T
    figure, axis = plt.subplots(figsize=(8, 5))
    values.plot(kind="bar", ax=axis)
    axis.set(xlabel="Reward", ylabel="Mean rate", title="Standard vs randomized reward rates")
    axis.tick_params(axis="x", rotation=0)
    return _save(figure, path)


def _model_domain_figures(metrics: dict, scatter_path: Path, gap_path: Path, k: int) -> list[str]:
    metric = f"ndcg@{k}"
    rows = []
    for model, values in metrics.items():
        if not isinstance(values, dict) or not {"standard", "randomized"}.issubset(values):
            continue
        standard = values["standard"].get(metric)
        randomized = values["randomized"].get(metric)
        if isinstance(standard, (int, float)) and isinstance(randomized, (int, float)):
            rows.append({"model": model, "standard": standard, "randomized": randomized})
    frame = pd.DataFrame(rows)
    if frame.empty:
        reason = f"No paired {metric} model metrics are available."
        return [
            _placeholder(scatter_path, "Model standard vs randomized performance", reason),
            _placeholder(gap_path, "Exposure generalization gap", reason),
        ]
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(frame["standard"], frame["randomized"], s=55)
    lower = float(frame[["standard", "randomized"]].min().min())
    upper = float(frame[["standard", "randomized"]].max().max())
    margin = max((upper - lower) * 0.08, 1e-4)
    axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], "--", color="gray")
    for row in frame.itertuples(index=False):
        axis.annotate(row.model, (row.standard, row.randomized), fontsize=8, xytext=(4, 4), textcoords="offset points")
    axis.set(xlabel=f"Standard {metric}", ylabel=f"Randomized {metric}", title="Model performance across exposure policies")
    scatter = _save(figure, scatter_path)

    gaps = frame.assign(gap=frame["standard"] - frame["randomized"]).sort_values("gap")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(gaps["model"], gaps["gap"])
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel=f"Standard minus randomized {metric}", title="Exposure generalization gap")
    return [scatter, _save(figure, gap_path)]


def _retrieval_recall(metrics: dict, path: Path) -> str:
    rows = []
    for domain in ("standard_test", "randomized_test_holdout"):
        values = metrics.get(domain, {})
        for name, value in values.items():
            if name.startswith("recall@") and isinstance(value, (int, float)):
                rows.append({"domain": domain, "k": int(name.split("@", 1)[1]), "recall": value})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _placeholder(path, "Retrieval recall at K", "No retrieval recall metrics are available.")
    figure, axis = plt.subplots(figsize=(8, 5))
    for domain, values in frame.groupby("domain", sort=True):
        values = values.sort_values("k")
        axis.plot(values["k"], values["recall"], marker="o", label=str(domain))
    axis.set(xlabel="K", ylabel="Recall", title="Two-Tower retrieval recall")
    axis.legend()
    return _save(figure, path)


def _ranker_ndcg(metrics: dict, path: Path) -> str:
    rows = []
    for model, domains in metrics.items():
        if not isinstance(domains, dict):
            continue
        for domain in ("standard", "randomized"):
            for name, value in domains.get(domain, {}).items():
                if name.startswith("ndcg@") and isinstance(value, (int, float)):
                    rows.append({"model": model, "domain": domain, "k": int(name.split("@", 1)[1]), "ndcg": value})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _placeholder(path, "Ranker NDCG at K", "No ranker NDCG metrics are available.")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, domain in zip(axes, ("standard", "randomized")):
        for model, values in frame.loc[frame["domain"].eq(domain)].groupby("model", sort=True):
            values = values.sort_values("k")
            axis.plot(values["k"], values["ndcg"], marker="o", label=str(model))
        axis.set(xlabel="K", ylabel="NDCG", title=domain.title())
    axes[-1].legend(fontsize=7, loc="best")
    figure.suptitle("Retrieval and ranker NDCG")
    return _save(figure, path)


def _cohort_share(root: Path, path: Path, k: int) -> str:
    rows = []
    for source in sorted((root / "outputs/metrics").glob("*_item_cohorts.parquet")):
        frame = pd.read_parquet(source)
        column = f"recommendation_share@{k}"
        if column not in frame:
            continue
        prefix = source.stem.removesuffix("_item_cohorts")
        domain = "randomized" if prefix.endswith("_randomized") else "standard"
        model = prefix.removesuffix(f"_{domain}")
        for _, row in frame.iterrows():
            rows.append({"model": model, "domain": domain, "cohort": str(row["cohort"]), "share": float(row[column])})
    values = pd.DataFrame(rows)
    if values.empty:
        return _placeholder(path, "Popularity-bucket recommendation share", "No item-cohort artifacts are available.")
    selected = values.loc[values["domain"].eq("randomized")]
    if selected.empty:
        selected = values
    pivot = selected.pivot_table(index="model", columns="cohort", values="share", aggfunc="mean").fillna(0)
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=axis)
    axis.set(xlabel="Model", ylabel=f"Recommendation share at {k}", title="Popularity-bucket recommendation mix")
    axis.tick_params(axis="x", rotation=30)
    return _save(figure, path)


def _coverage_gini(analysis: dict, path: Path) -> str:
    rows = []
    for model, domains in analysis.get("models", {}).items():
        for domain, values in domains.items():
            behavior = values.get("behavior", {})
            if {"catalog_coverage", "recommendation_gini"}.issubset(behavior):
                rows.append({"model": model, "domain": domain, **behavior})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _placeholder(path, "Catalog coverage and recommendation Gini", "No recommendation-behavior metrics are available.")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, metric, title in zip(
        axes,
        ("catalog_coverage", "recommendation_gini"),
        ("Catalog coverage", "Recommendation concentration (Gini)"),
    ):
        frame.pivot(index="model", columns="domain", values=metric).plot(kind="bar", ax=axis)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
    return _save(figure, path)


def _user_cohorts(root: Path, path: Path, k: int) -> str:
    rows = []
    column = f"ndcg@{k}"
    for source in sorted((root / "outputs/metrics").glob("*_user_cohorts.parquet")):
        frame = pd.read_parquet(source)
        if column not in frame:
            continue
        prefix = source.stem.removesuffix("_user_cohorts")
        domain = "randomized" if prefix.endswith("_randomized") else "standard"
        model = prefix.removesuffix(f"_{domain}")
        for _, row in frame.iterrows():
            rows.append({"model": model, "domain": domain, "cohort": str(row["cohort"]), "ndcg": float(row[column])})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _placeholder(path, "User-history cohort performance", "No user-cohort artifacts are available.")
    selected = frame.loc[frame["domain"].eq("randomized")]
    if selected.empty:
        selected = frame
    pivot = selected.pivot_table(index="model", columns="cohort", values="ndcg", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=axis)
    axis.set(xlabel="Model", ylabel=f"NDCG@{k}", title="Performance by user-history cohort")
    axis.tick_params(axis="x", rotation=30)
    return _save(figure, path)


def _weight_distribution(root: Path, path: Path) -> str:
    source = root / "outputs/metrics/training_weight_sample.parquet"
    if not source.is_file():
        return _placeholder(path, "Importance-weight distribution", "No sampled training-weight artifact is available.")
    frame = pd.read_parquet(source)
    columns = [value for value in ("density_raw", "density_selected", "inverse_sqrt_popularity") if value in frame]
    if not columns:
        return _placeholder(path, "Importance-weight distribution", "The training-weight artifact has no weight columns.")
    figure, axis = plt.subplots(figsize=(9, 5))
    plotted = 0
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if values.empty:
            continue
        values = values.clip(upper=float(values.quantile(0.995)))
        axis.hist(values, bins=80, density=True, alpha=0.4, label=column)
        plotted += 1
    if not plotted:
        plt.close(figure)
        return _placeholder(path, "Importance-weight distribution", "All sampled training weights are missing or non-finite.")
    axis.set(xlabel="Weight (capped at each series' 99.5th percentile)", ylabel="Density", title="Training-weight distributions")
    axis.legend()
    return _save(figure, path)


def _clip_selection(density: dict, path: Path) -> str:
    frame = pd.DataFrame(density.get("clip_selection", []))
    if frame.empty or not {"clip", "selection_value"}.issubset(frame):
        return _placeholder(path, "Randomized NDCG vs clipping", "No density-ratio clipping sweep is available.")
    frame = frame.dropna(subset=["selection_value"]).sort_values("clip")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame["clip"], frame["selection_value"], marker="o")
    selected = density.get("selected_clip")
    if selected is not None:
        axis.axvline(float(selected), linestyle="--", color="red", label=f"selected={selected}")
        axis.legend()
    metric = frame.get("selection_metric", pd.Series(["randomized adaptation NDCG"])).iloc[0]
    axis.set(xlabel="Maximum density-ratio weight", ylabel=str(metric), title="Clipping selected on randomized adaptation data")
    return _save(figure, path)


def _calibration_frontier(analysis: dict, path: Path, k: int) -> str:
    rows = []
    for domain, weights in analysis.get("calibration_diversity_frontier", {}).items():
        for weight, values in weights.items():
            ndcg = values.get("ranking", {}).get(f"ndcg@{k}")
            divergence = values.get("calibration", {}).get("js_divergence_mean")
            if isinstance(ndcg, (int, float)) and isinstance(divergence, (int, float)):
                rows.append({"domain": domain, "weight": float(weight), "ndcg": ndcg, "divergence": divergence})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _placeholder(path, "NDCG vs calibration divergence", "No calibration-reranking frontier is available.")
    figure, axis = plt.subplots(figsize=(8, 5))
    for domain, values in frame.groupby("domain", sort=True):
        values = values.sort_values("weight")
        axis.plot(values["divergence"], values["ndcg"], marker="o", label=str(domain))
        for row in values.itertuples(index=False):
            axis.annotate(f"w={row.weight:g}", (row.divergence, row.ndcg), fontsize=7)
    axis.set(xlabel="Jensen-Shannon divergence (lower is better)", ylabel=f"NDCG@{k}", title="Relevance-calibration trade-off")
    axis.legend()
    return _save(figure, path)


def _faiss_tradeoff(scaling: dict, path: Path) -> str:
    frame = pd.DataFrame(scaling.get("hnsw", {}).get("tradeoff", []))
    if frame.empty or not {"recall_at_k", "p95_ms"}.issubset(frame):
        reason = scaling.get("hnsw", {}).get("reason", "No HNSW scale benchmark is available.")
        return _placeholder(path, "FAISS recall-latency trade-off", str(reason))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame["p95_ms"], frame["recall_at_k"], marker="o")
    for row in frame.itertuples(index=False):
        axis.annotate(f"ef={row.ef_search}", (row.p95_ms, row.recall_at_k), fontsize=8)
    axis.set(xlabel="Per-query p95 latency (ms)", ylabel="Recall at candidate K", title="FAISS HNSW recall-latency trade-off")
    return _save(figure, path)


def _ope_sample_size(ope: dict, path: Path) -> str:
    rows = []
    for experiment in ope.get("experiments", []):
        sample_size = experiment.get("evaluation_rounds", experiment.get("sample_size"))
        for estimator in ("dm", "ips", "snips", "dr"):
            error = experiment.get("estimates", {}).get(f"{estimator}_absolute_error")
            if isinstance(sample_size, (int, float)) and isinstance(error, (int, float)):
                rows.append({"sample_size": sample_size, "estimator": estimator.upper(), "error": error})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _placeholder(path, "OPE error vs sample size", "OPE was not run because a propensity-bearing Open Bandit Dataset was not configured.")
    summary = frame.groupby(["sample_size", "estimator"], as_index=False)["error"].mean()
    figure, axis = plt.subplots(figsize=(8, 5))
    for estimator, values in summary.groupby("estimator", sort=True):
        values = values.sort_values("sample_size")
        axis.plot(values["sample_size"], values["error"], marker="o", label=str(estimator))
    axis.set(xlabel="Evaluation rounds", ylabel="Mean absolute error from on-policy value", title="OPE estimator error vs sample size")
    axis.legend()
    return _save(figure, path)


def generate_final_figures(root: Path, interactions: pd.DataFrame, config: dict) -> list[str]:
    """Generate the complete required figure set from durable experiment artifacts.

    Optional experiments receive an explicit placeholder figure when they were
    not configured, so a report never silently implies that an experiment ran.
    """
    output = root / "outputs/figures/final"
    rankers = _json(root / "outputs/metrics/rankers.json")
    retrieval = _json(root / "outputs/metrics/retrieval.json")
    analysis = _json(root / "outputs/metrics/robustness_and_calibration.json")
    density = _json(root / "outputs/metrics/density_ratio.json")
    scaling = _json(root / "outputs/metrics/scaling.json")
    ope = _json(root / "outputs/metrics/ope.json")
    k = int(config["final_k"])

    artifacts = [
        _standard_random_exposure(interactions, output / "01_standard_vs_random_item_exposure.png"),
        _standard_random_rewards(interactions, output / "02_standard_vs_random_reward_rates.png"),
    ]
    artifacts.extend(_model_domain_figures(
        rankers,
        output / "03_model_standard_vs_random_scatter.png",
        output / "04_exposure_generalization_gap.png",
        k,
    ))
    artifacts.extend([
        _retrieval_recall(retrieval, output / "05_retrieval_recall_at_k.png"),
        _ranker_ndcg(rankers, output / "06_ranker_ndcg_at_k.png"),
        _cohort_share(root, output / "07_popularity_bucket_recommendation_share.png", k),
        _coverage_gini(analysis, output / "08_catalog_coverage_gini.png"),
        _user_cohorts(root, output / "09_user_history_cohort_performance.png", k),
        _weight_distribution(root, output / "10_importance_weight_distribution.png"),
        _clip_selection(density, output / "11_randomized_ndcg_vs_clipping.png"),
        _calibration_frontier(analysis, output / "12_ndcg_vs_calibration_divergence.png", k),
        _faiss_tradeoff(scaling, output / "13_faiss_recall_latency.png"),
        _ope_sample_size(ope, output / "14_ope_error_vs_sample_size.png"),
    ])
    return [str(Path(value).relative_to(root)) for value in artifacts]
