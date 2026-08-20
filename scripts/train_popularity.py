from __future__ import annotations

import sys

from ranklab.data.kuairand import load_interactions
from ranklab.data.splitting import assign_splits, validate_splits
from ranklab.retrieval.experiment import evaluate_and_save, save_model
from ranklab.retrieval.popularity import fit_popularity
from ranklab.utils.config import experiment_config


def main() -> None:
    config = experiment_config(sys.argv[1:], "configs/retrieval/popularity.yaml")
    interactions = assign_splits(load_interactions(config["raw_dir"]), config)
    validate_splits(interactions)
    train = interactions.loc[interactions["split"].eq("train")]
    model = fit_popularity(train, alpha=float(config["alpha"]))
    save_model(model, "popularity", config)
    print(evaluate_and_save(model, interactions, "popularity", config))


if __name__ == "__main__":
    main()
