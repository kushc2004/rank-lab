from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ranklab.data.kuairand import discovered_file_summary, load_interactions, load_side_tables
from ranklab.data.splitting import assign_splits, validate_splits, write_manifests
from ranklab.utils.config import kuairand_config


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def _feature_audit(interactions: pd.DataFrame, side_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for column in interactions.columns:
        if column in {"user_id", "item_id"}:
            status, reason = "identifier only", "Used only as a key; never supplied as a numeric feature."
        elif column in {"long_view", "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time", "is_profile_enter"}:
            status, reason = "excluded", "Interaction outcome or post-exposure behaviour."
        elif column == "is_random":
            status, reason = "excluded", "Logging-policy indicator; randomized rows are never training data."
        elif column in {"timestamp_ms", "date", "tab"}:
            status, reason = "allowed exposure-time context", "Known at exposure time; timestamp/date also define strict temporal splits and histories."
        elif column == "hourmin":
            status, reason = "excluded redundant context", "Timestamp-derived hour and day-of-week are used instead."
        else:
            status, reason = "excluded", "Source bookkeeping field."
        rows.append({"table": "interactions", "column": column, "status": status, "reason": reason})
    for table, frame in side_tables.items():
        for column in frame.columns:
            if column in {"user_id", "video_id", "item_id"}:
                status = "identifier only"
                reason = "Used as a join key, never as a scalar numeric feature."
            elif column == "tag":
                status = "analysis only"
                reason = "Used only for post-hoc calibration analysis; never supplied to a trained primary model."
            else:
                status = "excluded from primary models"
                reason = "Official release does not establish a per-row historical availability time; available only behind the relaxed side-feature ablation."
            rows.append({
                "table": table,
                "column": column,
                "status": status,
                "reason": reason,
            })
    return pd.DataFrame(rows)


def _plots(interactions: pd.DataFrame, feature_audit: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    standard = interactions.loc[interactions["is_random"].eq(0)]
    random = interactions.loc[interactions["is_random"].eq(1)]
    figure, axis = plt.subplots()
    interactions.groupby("date").size().plot(ax=axis)
    axis.set(title="Interactions by day", ylabel="rows")
    figure.tight_layout(); figure.savefig(output_dir / "interactions_by_day.png"); plt.close(figure)

    rates = interactions.groupby(["date", "source_log"])["long_view"].mean().unstack()
    figure, axis = plt.subplots()
    rates.plot(ax=axis)
    axis.set(title="Long-view reward rate by day and log", ylabel="rate")
    figure.tight_layout(); figure.savefig(output_dir / "reward_rates.png"); plt.close(figure)

    user_activity = pd.DataFrame({
        "standard": standard.groupby("user_id").size(),
        "randomized": random.groupby("user_id").size(),
    }).fillna(0)
    figure, axis = plt.subplots()
    for column in user_activity:
        axis.hist(np.log1p(user_activity[column]), bins=40, alpha=0.55, label=column)
    axis.set(title="User activity distribution", xlabel="log(1 + interactions)", ylabel="users")
    axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "user_activity_distribution.png"); plt.close(figure)

    item_activity = pd.DataFrame({
        "standard": standard.groupby("item_id").size(),
        "randomized": random.groupby("item_id").size(),
    }).fillna(0)
    figure, axis = plt.subplots()
    for column in item_activity:
        axis.hist(np.log1p(item_activity[column]), bins=50, alpha=0.55, label=column)
    axis.set(title="Item exposure distribution", xlabel="log(1 + exposures)", ylabel="items")
    axis.legend(); figure.tight_layout(); figure.savefig(output_dir / "item_exposure_distribution.png"); plt.close(figure)

    figure, axis = plt.subplots()
    axis.scatter(
        item_activity["standard"] + 1, item_activity["randomized"] + 1,
        alpha=0.25, s=10,
    )
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set(title="Standard vs randomized item frequency", xlabel="standard exposures + 1", ylabel="randomized exposures + 1")
    figure.tight_layout(); figure.savefig(output_dir / "standard_vs_random_item_frequency.png"); plt.close(figure)

    figure, axis = plt.subplots()
    interactions.groupby(["tab", "source_log"]).size().unstack(fill_value=0).plot(kind="bar", ax=axis)
    axis.set(title="Tab distribution", ylabel="rows")
    figure.tight_layout(); figure.savefig(output_dir / "tab_distribution.png"); plt.close(figure)

    shift = pd.DataFrame({
        "standard": standard.groupby("tab")["long_view"].mean(),
        "randomized": random.groupby("tab")["long_view"].mean(),
    }).fillna(0)
    figure, axis = plt.subplots()
    shift.plot(kind="bar", ax=axis)
    axis.set(title="Standard vs randomized reward shift by tab", ylabel="long-view rate")
    figure.tight_layout(); figure.savefig(output_dir / "standard_vs_random_feature_shift.png"); plt.close(figure)


def main() -> None:
    config = kuairand_config(sys.argv[1:])
    interactions = assign_splits(load_interactions(config["raw_dir"]), config)
    validate_splits(interactions)
    side_tables = load_side_tables(config["raw_dir"])
    standard = interactions.loc[interactions["is_random"].eq(0)]
    random = interactions.loc[interactions["is_random"].eq(1)]
    file_summary = pd.DataFrame(discovered_file_summary(config["raw_dir"]))
    split_counts = write_manifests(interactions, config)
    feature_audit = _feature_audit(interactions, side_tables)
    reports = Path(config["reports_dir"]); reports.mkdir(parents=True, exist_ok=True)
    feature_audit.to_csv(reports / "feature_leakage_audit.csv", index=False)
    summary = interactions.groupby(["source_log", "is_random"]).agg(rows=("item_id", "size"), users=("user_id", "nunique"), items=("item_id", "nunique"), min_date=("date", "min"), max_date=("date", "max")).reset_index()
    dtypes = pd.DataFrame({"column": interactions.columns, "dtype": interactions.dtypes.astype(str).values})
    table_schemas = []
    for table, frame in {"interactions": interactions, **side_tables}.items():
        table_schemas.extend(
            {
                "table": table,
                "rows": len(frame),
                "column": column,
                "dtype": str(frame[column].dtype),
                "missing": int(frame[column].isna().sum()),
            }
            for column in frame.columns
        )
    schema_frame = pd.DataFrame(table_schemas)
    duplicate_keys = int(interactions.duplicated(
        ["source_log", "user_id", "item_id", "timestamp_ms"], keep=False
    ).sum())
    rewards = interactions.groupby(["source_log", "is_random"]).agg(
        rows=("item_id", "size"), long_view_rate=("long_view", "mean"),
        click_rate=("is_click", "mean"), like_rate=("is_like", "mean"),
    ).reset_index()
    policy_overlap = pd.DataFrame([
        {
            "entity": entity,
            "standard_unique": int(standard_count),
            "randomized_unique": int(random_count),
            "intersection": int(overlap),
        }
        for entity, standard_count, random_count, overlap in (
            (
                "users", standard["user_id"].nunique(), random["user_id"].nunique(),
                len(set(standard["user_id"]) & set(random["user_id"])),
            ),
            (
                "items", standard["item_id"].nunique(), random["item_id"].nunique(),
                len(set(standard["item_id"]) & set(random["item_id"])),
            ),
        )
    ])
    report = [
        "# KuaiRand-Pure reproducible data audit", "",
        "## Official files discovered", "", _markdown_table(file_summary.drop(columns="columns")), "",
        "## Canonical interaction schema", "", _markdown_table(dtypes), "",
        "## All table schemas, dtypes, and missing values", "", _markdown_table(schema_frame), "",
        "## Interaction coverage", "", _markdown_table(summary), "",
        "## Data quality", "", f"Duplicate canonical interaction keys (all copies counted): {duplicate_keys}.", "",
        _markdown_table(rewards), "",
        "## Standard/randomized entity overlap", "", _markdown_table(policy_overlap), "",
        "## Split manifest", "", _markdown_table(pd.DataFrame([split_counts])), "",
        f"Split boundaries: standard train through {config['train_end']}; validation "
        f"{config['validation_start']}-{config['validation_end']}; standard test "
        f"{config['standard_test_start']}-{config['standard_test_end']}; randomized rows are evaluation-only, "
        f"with adaptation through {config.get('randomized_validation_end', config['validation_end'])} and held-out evaluation from "
        f"{config.get('randomized_evaluation_start', config['standard_test_start'])}.", "",
        "## Candidate feature leakage audit", "", _markdown_table(feature_audit), "",
    ]
    (reports / "kuairand_data_audit.md").write_text("\n".join(report))
    (reports / "kuairand_data_audit.json").write_text(json.dumps({
        "files": file_summary.to_dict("records"),
        "table_schemas": schema_frame.to_dict("records"),
        "interaction_coverage": summary.to_dict("records"),
        "duplicate_canonical_keys": duplicate_keys,
        "reward_rates": rewards.to_dict("records"),
        "policy_overlap": policy_overlap.to_dict("records"),
        "split_counts": split_counts,
    }, indent=2) + "\n")
    _plots(interactions, feature_audit, Path(config["figures_dir"]))
    print((reports / "kuairand_data_audit.md").resolve())


if __name__ == "__main__":
    main()
