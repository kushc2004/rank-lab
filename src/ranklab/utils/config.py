from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]

def load_yaml(relative: str) -> dict:
    with (ROOT / relative).open() as handle:
        return yaml.safe_load(handle)

def parse_overrides(argv: list[str]) -> dict:
    return dict(token.split("=", 1) for token in argv if "=" in token)

def kuairand_config(argv: list[str]) -> dict:
    config = load_yaml("configs/data/kuairand_pure.yaml")
    config.update(parse_overrides(argv))
    return config


def experiment_config(argv: list[str], retrieval_config: str) -> dict:
    config = kuairand_config(argv)
    config.update(load_yaml(retrieval_config))
    config.update(load_yaml("configs/experiment/exposure_gap.yaml"))
    config.update(parse_overrides(argv))
    return config
