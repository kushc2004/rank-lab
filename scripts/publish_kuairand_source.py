#!/usr/bin/env python3
"""Publish the unmodified, checksum-verified official archive as a Kaggle input."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/KuaiRand-Pure.tar.gz"
EXPECTED_MD5 = "0820331067a3784d9691136f772b35a7"
DEFAULT_DATASET = "kushchaudhari/kuairand-pure-official"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    arguments = parser.parse_args()
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Official archive not found: {ARCHIVE}")
    if _md5(ARCHIVE) != EXPECTED_MD5:
        raise ValueError("KuaiRand-Pure archive checksum does not match the official release")

    staging = ROOT / "artifacts/kuairand-source"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copy2(ARCHIVE, staging / ARCHIVE.name)
    (staging / "dataset-metadata.json").write_text(json.dumps({
        "title": "KuaiRand-Pure Official Archive",
        "id": arguments.dataset,
        "licenses": [{"name": "CC-BY-SA-4.0"}],
    }, indent=2) + "\n")
    print(f"staged verified {ARCHIVE.name} for {arguments.dataset}")
    if arguments.no_upload:
        return
    command = ["kaggle", "datasets", "create" if arguments.create else "version", "-p", str(staging)]
    if not arguments.create:
        command.extend(["-m", "Refresh verified official KuaiRand-Pure archive"])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
