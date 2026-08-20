from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from ranklab.evaluation.ranking_metrics import evaluate_ranked_impressions


def evaluate_and_save(model, interactions: pd.DataFrame, model_name: str, config: dict) -> dict:
    metrics_dir = Path(config["metrics_dir"])
    predictions_dir = Path(config["predictions_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict] = {}
    for label, split in (("standard", "standard_test"), ("randomized", "randomized_test")):
        frame = interactions.loc[interactions["split"].eq(split)].copy()
        frame["score"] = model.predict(frame["user_id"], frame["item_id"])
        metrics, per_group = evaluate_ranked_impressions(
            frame,
            k_values=tuple(config["k_values"]),
            min_group_size=int(config["min_group_size"]),
        )
        frame.to_parquet(predictions_dir / f"{model_name}_{label}.parquet", index=False)
        per_group.to_parquet(predictions_dir / f"{model_name}_{label}_per_group.parquet", index=False)
        result[label] = metrics
    (metrics_dir / f"{model_name}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def save_model(model, model_name: str, config: dict) -> Path:
    path = Path(config["models_dir"]) / f"{model_name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        pickle.dump(model, output)
    return path
