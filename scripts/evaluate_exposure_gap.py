from __future__ import annotations

import sys
from pathlib import Path

from ranklab.reporting.exposure_gap import write_exposure_gap
from ranklab.utils.config import kuairand_config


def main() -> None:
    config = kuairand_config(sys.argv[1:])
    report = Path(config["reports_dir"]) / "initial_exposure_gap.md"
    write_exposure_gap(Path(config["metrics_dir"]), report)
    print(report.resolve())


if __name__ == "__main__":
    main()
