from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]

def load_yaml(relative: str) -> dict:
    with (ROOT / relative).open() as handle:
        return yaml.safe_load(handle)

def parse_overrides(argv: list[str]) -> dict:
    overrides: dict[str, object] = {}
    for token in argv:
        if "=" not in token:
            continue
        key, raw = token.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    return overrides

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
