from __future__ import annotations

import sys

from ranklab.data.kuairand import load_interactions
from ranklab.data.splitting import assign_splits, validate_splits
from ranklab.retrieval.bpr import fit_bpr
from ranklab.retrieval.experiment import evaluate_and_save, save_model
from ranklab.utils.config import experiment_config


def main() -> None:
    config = experiment_config(sys.argv[1:], "configs/retrieval/bpr_mf.yaml")
    interactions = assign_splits(load_interactions(config["raw_dir"]), config)
    validate_splits(interactions)
    train = interactions.loc[interactions["split"].eq("train")]
    model = fit_bpr(
        train,
        embedding_dim=int(config["embedding_dim"]),
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        regularization=float(config["regularization"]),
        seed=int(config["seed"]),
        device=str(config["device"]),
        batch_size=int(config["batch_size"]),
    )
    save_model(model, "bpr", config)
    print(evaluate_and_save(model, interactions, "bpr", config))


if __name__ == "__main__":
    main()
