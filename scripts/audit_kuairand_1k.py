#!/usr/bin/env python3
"""Create a streaming, reproducible audit of an attached official KuaiRand-1K release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ranklab.data.kuairand_1k import (
    OFFICIAL_1K_ARCHIVE,
    OFFICIAL_1K_MD5,
    OFFICIAL_1K_URL,
    discovered_file_summary,
    streaming_interaction_audit,
)


def markdown(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="official KuaiRand-1K extraction root or its data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/kuairand_1k"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    files = discovered_file_summary(args.raw_dir)
    interactions = streaming_interaction_audit(args.raw_dir, chunksize=args.chunksize)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "KuaiRand-1K",
        "official_source": OFFICIAL_1K_URL,
        "official_archive": OFFICIAL_1K_ARCHIVE,
        "official_md5": OFFICIAL_1K_MD5,
        "raw_dir": str(args.raw_dir.resolve()),
        "files": files,
        "interactions": interactions,
        "audit_contract": (
            "Interaction counts, date ranges, timestamp ranges, and per-log unique "
            "users/items are exact streamed values. CSV dtypes are intentionally not "
            "claimed from headers alone; a training adapter must declare parse dtypes."
        ),
    }
    (args.output_dir / "kuairand_1k_data_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# KuaiRand-1K audit", "",
        "This is a separate official corpus, not a replacement for the KuaiRand-Pure exposure benchmark.", "",
        "## Source contract", "",
        f"- Official archive: `{OFFICIAL_1K_ARCHIVE}`",
        f"- Official MD5: `{OFFICIAL_1K_MD5}`",
        f"- Official URL: {OFFICIAL_1K_URL}", "",
        "## Discovered files", "", markdown(pd.DataFrame(files).drop(columns=["columns", "sample_dtypes"])), "",
        "## Exact streamed interaction audit", "", markdown(pd.DataFrame(interactions)), "",
    ]
    (args.output_dir / "kuairand_1k_data_audit.md").write_text("\n".join(report))
    print((args.output_dir / "kuairand_1k_data_audit.md").resolve())


if __name__ == "__main__":
    main()
