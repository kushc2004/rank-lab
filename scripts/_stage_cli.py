#!/usr/bin/env python3
"""Shared implementation for the specification's single-stage commands."""
from __future__ import annotations

import argparse
import re

import yaml

from run_full_pipeline import FullPipeline, ROOT, load_config


PROFILE_GROUPS = {
    "data": "data",
    "retrieval": "retrieval",
    "ranking": "ranking",
    "debias": "debias",
    "rerank": "rerank",
    "experiment": "experiment",
}
IDENTIFIER_KEYS = {"run_id", "retrieval_run_id", "run_ids"}
SAFE_PROFILE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _normalized_overrides(values: list[str]) -> list[str]:
    """Translate the documented Hydra-style selectors to the unified config.

    RankLab has one durable experiment state rather than disconnected run IDs;
    selectors and run IDs therefore name the contract but do not alter model
    parameters. Actual scalar/path overrides continue to flow to the config.
    """
    normalized: list[str] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must be key=value, received {value!r}")
        key, raw = value.split("=", 1)
        if key in PROFILE_GROUPS or key in IDENTIFIER_KEYS:
            continue
        if key == "top_k":
            normalized.append(f"candidate_k={raw}")
        else:
            normalized.append(value)
    return normalized


def _load_stage_config(values: list[str]) -> dict:
    config = load_config([])
    selectors: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must be key=value, received {value!r}")
        key, raw = value.split("=", 1)
        if key in PROFILE_GROUPS:
            profile = str(yaml.safe_load(raw))
            if not SAFE_PROFILE.fullmatch(profile):
                raise ValueError(f"invalid {key} profile name: {profile!r}")
            selectors[key] = profile

    for key in ("data", "retrieval", "ranking", "debias", "rerank", "experiment"):
        profile = selectors.get(key)
        if profile is None:
            continue
        path = ROOT / "configs" / PROFILE_GROUPS[key] / f"{profile}.yaml"
        if not path.is_file():
            raise ValueError(f"unknown {key} profile {profile!r}: {path}")
        with path.open(encoding="utf-8") as source:
            config.update(yaml.safe_load(source) or {})

    for value in _normalized_overrides(values):
        key, raw = value.split("=", 1)
        config[key] = yaml.safe_load(raw)
    return config


def run_stage(stage: str, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("overrides", nargs="*", help="configuration overrides as key=value")
    args = parser.parse_args()
    pipeline = FullPipeline(
        ROOT,
        _load_stage_config(args.overrides),
        force=args.force,
    )
    pipeline.run(stage, stage)
    print((ROOT / "outputs/full_pipeline_state.json").resolve())
