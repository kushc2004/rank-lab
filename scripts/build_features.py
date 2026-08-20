from __future__ import annotations

import json
import sys
from pathlib import Path

from ranklab.data.kuairand import load_interactions
from ranklab.data.splitting import assign_splits, validate_splits
from ranklab.features.historical import build_historical_features
from ranklab.utils.config import kuairand_config


def main() -> None:
    config = kuairand_config(sys.argv[1:])
    interactions = assign_splits(load_interactions(config["raw_dir"]), config)
    validate_splits(interactions)
    train = interactions.loc[interactions["split"].eq("train")].copy()
    if train["is_random"].any():
        raise AssertionError("Randomized logs cannot enter historical-feature training data")
    features = build_historical_features(train, alpha=float(config["feature_alpha"]))
    if not (features["feature_cutoff_ms"] < features["timestamp_ms"]).all():
        raise AssertionError("A feature cutoff is not strictly before its row timestamp")
    directory = Path(config["features_dir"]); directory.mkdir(parents=True, exist_ok=True)
    features.to_parquet(directory / "train_historical.parquet", index=False)
    (directory / "feature_manifest.json").write_text(json.dumps({"source_split": "train", "randomized_rows": 0, "rows": len(features), "cutoff_rule": "strictly earlier timestamp"}, indent=2) + "\n")
    print((directory / "train_historical.parquet").resolve())


if __name__ == "__main__":
    main()
