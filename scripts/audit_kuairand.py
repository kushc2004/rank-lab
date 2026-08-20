from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
        elif column in {"timestamp_ms", "date", "hourmin", "tab"}:
            status, reason = "not used in M0-M3", "Context/provenance field; only timestamp is used for ordering and split assignment."
        else:
            status, reason = "excluded", "Source bookkeeping field."
        rows.append({"table": "interactions", "column": column, "status": status, "reason": reason})
    for table, frame in side_tables.items():
        for column in frame.columns:
            rows.append({
                "table": table,
                "column": column,
                "status": "excluded pending as-of provenance",
                "reason": "Official release does not establish a per-row historical availability time; M0-M3 uses only reconstructed histories.",
            })
    return pd.DataFrame(rows)


def _plots(interactions: pd.DataFrame, feature_audit: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    standard = interactions.loc[interactions["is_random"].eq(0)]
    random = interactions.loc[interactions["is_random"].eq(1)]
    plt.figure(); interactions.groupby("date").size().plot(); plt.title("Daily interaction volume"); plt.ylabel("rows"); plt.tight_layout(); plt.savefig(output_dir / "daily_activity.png"); plt.close()
    plt.figure(); interactions.groupby("date")["long_view"].mean().plot(); plt.title("Daily long-view rate"); plt.ylabel("rate"); plt.tight_layout(); plt.savefig(output_dir / "daily_reward_rate.png"); plt.close()
    plt.figure(); pd.concat([standard.groupby("user_id").size(), random.groupby("user_id").size()]).value_counts().sort_index().head(50).plot(kind="bar"); plt.title("User activity distribution (first 50 counts)"); plt.tight_layout(); plt.savefig(output_dir / "user_activity_distribution.png"); plt.close()
    plt.figure(); pd.concat([standard.groupby("item_id").size(), random.groupby("item_id").size()]).value_counts().sort_index().head(50).plot(kind="bar"); plt.title("Item activity distribution (first 50 counts)"); plt.tight_layout(); plt.savefig(output_dir / "item_activity_distribution.png"); plt.close()
    plt.figure(); interactions.groupby("tab").size().plot(kind="bar"); plt.title("Tab distribution"); plt.tight_layout(); plt.savefig(output_dir / "tab_distribution.png"); plt.close()
    shift = pd.DataFrame({"standard": standard.groupby("tab")["long_view"].mean(), "randomized": random.groupby("tab")["long_view"].mean()}).fillna(0)
    plt.figure(); shift.plot(kind="bar"); plt.title("Standard vs randomized feature shift by tab"); plt.ylabel("long-view rate"); plt.tight_layout(); plt.savefig(output_dir / "standard_vs_random_feature_shift.png"); plt.close()


def main() -> None:
    config = kuairand_config(sys.argv[1:])
    interactions = assign_splits(load_interactions(config["raw_dir"]), config)
    validate_splits(interactions)
    side_tables = load_side_tables(config["raw_dir"])
    file_summary = pd.DataFrame(discovered_file_summary(config["raw_dir"]))
    split_counts = write_manifests(interactions, config)
    feature_audit = _feature_audit(interactions, side_tables)
    reports = Path(config["reports_dir"]); reports.mkdir(parents=True, exist_ok=True)
    feature_audit.to_csv(reports / "feature_leakage_audit.csv", index=False)
    summary = interactions.groupby(["source_log", "is_random"]).agg(rows=("item_id", "size"), users=("user_id", "nunique"), items=("item_id", "nunique"), min_date=("date", "min"), max_date=("date", "max")).reset_index()
    dtypes = pd.DataFrame({"column": interactions.columns, "dtype": interactions.dtypes.astype(str).values})
    report = ["# KuaiRand-Pure reproducible data audit", "", "## Official files discovered", "", _markdown_table(file_summary.drop(columns="columns")), "", "## Canonical interaction schema", "", _markdown_table(dtypes), "", "## Interaction coverage", "", _markdown_table(summary), "", "## Split manifest", "", _markdown_table(pd.DataFrame([split_counts])), "", "Split boundaries: standard train through 20220421; validation 20220422-20220430; standard test 20220501-20220508; all randomized dates are randomized test.", "", "## Candidate feature leakage audit", "", _markdown_table(feature_audit), ""]
    (reports / "kuairand_data_audit.md").write_text("\n".join(report))
    (reports / "kuairand_data_audit.json").write_text(json.dumps({"files": file_summary.to_dict("records"), "split_counts": split_counts}, indent=2) + "\n")
    _plots(interactions, feature_audit, Path(config["figures_dir"]))
    print((reports / "kuairand_data_audit.md").resolve())


if __name__ == "__main__":
    main()
