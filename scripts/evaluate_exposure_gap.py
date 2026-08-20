from __future__ import annotations

import sys
from pathlib import Path

from _stage_cli import run_stage
from ranklab.reporting.exposure_gap import write_exposure_gap
from ranklab.utils.config import kuairand_config


def main() -> None:
    if any(argument.startswith("run_ids=") for argument in sys.argv[1:]):
        run_stage(
            "analysis",
            "Compare persisted retrieval/ranker runs across standard and randomized exposure.",
        )
        return
    config = kuairand_config(sys.argv[1:])
    report = Path(config["reports_dir"]) / "initial_exposure_gap.md"
    write_exposure_gap(Path(config["metrics_dir"]), report)
    print(report.resolve())


if __name__ == "__main__":
    main()
