#!/usr/bin/env python3
"""Run the complete RankLab specification with per-stage durable checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from ranklab.pipeline import FullPipeline


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("baselines", "two_tower", "candidates", "rankers", "evaluation", "analysis", "ope", "scale", "serving", "report")
CONFIGS = (
    "configs/data/kuairand_pure.yaml", "configs/experiment/exposure_gap.yaml",
    "configs/retrieval/two_tower.yaml", "configs/ranking/lgbm_lambdarank.yaml",
    "configs/debias/density_ratio.yaml", "configs/rerank/calibration.yaml",
    "configs/experiment/full_pipeline.yaml",
)


def load_config(overrides: list[str]) -> dict:
    config: dict = {}
    for relative in CONFIGS:
        with (ROOT / relative).open() as source:
            config.update(yaml.safe_load(source) or {})
    for value in overrides:
        if "=" not in value:
            raise ValueError(f"override must be key=value, received {value!r}")
        key, raw = value.split("=", 1)
        config[key] = yaml.safe_load(raw)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-stage", choices=STAGES)
    parser.add_argument("--to-stage", choices=STAGES)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("overrides", nargs="*", help="configuration overrides as key=value")
    args = parser.parse_args()
    pipeline = FullPipeline(ROOT, load_config(args.overrides), force=args.force)
    pipeline.run(args.from_stage, args.to_stage)
    print((ROOT / "outputs/full_pipeline_state.json").resolve())


if __name__ == "__main__":
    main()
