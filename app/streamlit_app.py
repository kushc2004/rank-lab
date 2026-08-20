from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs/metrics"
PREDICTIONS = ROOT / "outputs/predictions"
RUN = ROOT / "outputs/run"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def metric_table(payload: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for model, domains in payload.items():
        if not isinstance(domains, dict) or not {"standard", "randomized"}.issubset(domains):
            continue
        for metric, standard in domains["standard"].items():
            randomized = domains["randomized"].get(metric)
            if isinstance(standard, (int, float)) and isinstance(randomized, (int, float)):
                rows.append({
                    "model": model,
                    "metric": metric,
                    "standard": standard,
                    "randomized": randomized,
                    "standard_minus_randomized": standard - randomized,
                })
    return pd.DataFrame(rows)


st.set_page_config(page_title="RankLab", layout="wide")
st.title("RankLab: exposure-aware recommendation evaluation")
st.caption("Read-only view over checkpointed artifacts. This app never trains or mutates models.")

available = sorted(METRICS.glob("*.json"))
if not available:
    st.warning("No metrics found. Run scripts/run_full_pipeline.py first.")
    st.stop()

tabs = st.tabs([
    "Model comparison", "User inspector", "Bias explorer", "Debiasing",
    "Calibration and diversity", "OPE", "Scaling and latency", "Reproducibility",
])

with tabs[0]:
    rankers = read_json(METRICS / "rankers.json")
    comparison = metric_table(rankers)
    if comparison.empty:
        st.info("Ranker evaluation has not completed.")
    else:
        metric = st.selectbox("Metric", sorted(comparison["metric"].unique()))
        selected = comparison.loc[comparison["metric"].eq(metric)].copy()
        st.dataframe(selected, use_container_width=True, hide_index=True)
        st.bar_chart(selected.set_index("model")[["standard", "randomized"]])

with tabs[1]:
    prediction_files = sorted(
        path for path in PREDICTIONS.glob("*_standard.parquet")
        if "frontier" not in path.stem
    )
    if not prediction_files:
        st.info("No user-level predictions are available.")
    else:
        prediction_path = st.selectbox(
            "Predictions", prediction_files, format_func=lambda path: path.stem
        )
        predictions = pd.read_parquet(prediction_path)
        users = sorted(predictions["user_id"].dropna().astype(int).unique())
        user_id = st.selectbox("User", users)
        user_rows = predictions.loc[predictions["user_id"].eq(user_id)].copy()
        contexts = user_rows["context_id"].drop_duplicates().tolist()
        context_id = st.selectbox("Context", contexts)
        context = user_rows.loc[user_rows["context_id"].eq(context_id)].sort_values(
            ["score", "item_id"], ascending=[False, True]
        )
        columns = [
            value for value in (
                "item_id", "score", "retrieval_score", "label", "label_binary",
                "label_graded", "long_view", "item_exposure_count", "item_reward_rate",
                "user_history_count",
            ) if value in context
        ]
        st.dataframe(context[columns], use_container_width=True, hide_index=True)

with tabs[2]:
    cohort_files = sorted(METRICS.glob("*_cohorts.parquet"))
    if not cohort_files:
        st.info("Cohort analysis has not completed.")
    else:
        cohort_path = st.selectbox(
            "Cohort artifact", cohort_files, format_func=lambda path: path.stem
        )
        st.dataframe(pd.read_parquet(cohort_path), use_container_width=True, hide_index=True)
    robustness = read_json(METRICS / "robustness_and_calibration.json")
    if robustness.get("model_ranking_stability"):
        st.subheader("Standard vs randomized model-order stability")
        st.json(robustness["model_ranking_stability"])

with tabs[3]:
    density = read_json(METRICS / "density_ratio.json")
    if density:
        st.subheader("Density-ratio diagnostics and clipping selection")
        st.json(density)
    robustness = read_json(METRICS / "robustness_and_calibration.json")
    paired = robustness.get("paired_model_comparisons")
    if paired:
        st.subheader("Paired user-level bootstrap comparisons")
        st.json(paired)
    if not density and not paired:
        st.info("Debiasing artifacts have not completed.")

with tabs[4]:
    robustness = read_json(METRICS / "robustness_and_calibration.json")
    frontier = robustness.get("calibration_diversity_frontier", {})
    if not frontier:
        st.info("Calibration/diversity reranking has not completed.")
    else:
        domain = st.radio("Evaluation domain", ["standard", "randomized"], horizontal=True)
        rows = []
        for weight, values in frontier.get(domain, {}).items():
            row = {"relevance_weight": float(weight)}
            row.update({f"ranking_{key}": value for key, value in values.get("ranking", {}).items()})
            row.update({f"calibration_{key}": value for key, value in values.get("calibration", {}).items() if isinstance(value, (int, float))})
            row.update({f"diversity_{key}": value for key, value in values.get("diversity", {}).items() if isinstance(value, (int, float))})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    probability = read_json(METRICS / "probability_calibration.json")
    if probability:
        st.subheader("Pointwise probability calibration")
        st.json(probability)

with tabs[5]:
    ope = read_json(METRICS / "ope.json")
    if not ope:
        st.info("OPE is optional and requires an explicit Open Bandit Dataset input.")
    else:
        st.json(ope.get("evaluation_contract", {}))
        st.metric("On-policy random value", ope["on_policy_ground_truth"]["estimate"])
        rows = []
        for experiment in ope.get("experiments", []):
            for estimator in ("dm", "ips", "snips", "dr"):
                rows.append({
                    "seed": experiment["seed"],
                    "sample_fraction": experiment["sample_fraction"],
                    "clip": experiment["clip"],
                    "estimator": estimator,
                    "estimate": experiment["estimates"][estimator],
                    "absolute_error": experiment["estimates"][f"{estimator}_absolute_error"],
                    "effective_sample_size": experiment["importance_weights"]["effective_sample_size"],
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tabs[6]:
    scaling = read_json(METRICS / "scaling.json")
    latency = read_json(METRICS / "serving_latency.json")
    if scaling:
        st.subheader("Controlled exact-vs-HNSW study")
        st.caption(scaling.get("corpus_claim", ""))
        tradeoff = pd.DataFrame(scaling.get("hnsw", {}).get("tradeoff", []))
        st.dataframe(tradeoff, use_container_width=True, hide_index=True)
        if not tradeoff.empty:
            st.line_chart(tradeoff.set_index("ef_search")[["recall_at_k", "p95_ms"]])
        st.json({key: value for key, value in scaling.items() if key != "hnsw"})
    if latency:
        st.subheader("Local end-to-end serving benchmark")
        st.caption(latency.get("label", ""))
        stages = pd.DataFrame(latency.get("stages", {})).T.reset_index(names="stage")
        st.dataframe(stages, use_container_width=True, hide_index=True)
    if not scaling and not latency:
        st.info("Scaling and serving stages have not completed.")

with tabs[7]:
    reproducibility = read_json(RUN / "reproducibility.json")
    state = read_json(ROOT / "outputs/full_pipeline_state.json")
    st.subheader("Run identity")
    st.json(reproducibility or state)
    st.subheader("Data manifest")
    st.json(read_json(RUN / "data_manifest.json"))
    st.subheader("Split manifest")
    st.json(read_json(RUN / "split_manifest.json"))
    st.subheader("Feature contract")
    st.json(read_json(RUN / "feature_manifest.json"))
