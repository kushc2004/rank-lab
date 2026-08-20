#!/usr/bin/env python3
"""Package derived RankLab artifacts and optionally publish them with Kaggle CLI."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "kushchaudhari/ranklab-baseline-artifacts"
INCLUDE = ("data/manifests", "data/features", "data/processed", "data/indices", "outputs")


def _archive(staging: Path) -> Path:
    archive_path = staging / "ranklab_artifacts.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for relative in INCLUDE:
            directory = ROOT / relative
            if directory.is_dir():
                for file_path in sorted(path for path in directory.rglob("*") if path.is_file() and path.name != ".DS_Store"):
                    archive.add(file_path, arcname=file_path.relative_to(ROOT))
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--create", action="store_true", help="Create the dataset; otherwise publish a new version.")
    parser.add_argument("--public", action="store_true", help="Only applies with --create. Default is private.")
    parser.add_argument("--message", default="Refresh RankLab derived experiment artifacts")
    parser.add_argument("--no-upload", action="store_true", help="Package only, for a Kaggle notebook output.")
    arguments = parser.parse_args()

    artifacts_root = ROOT / "artifacts"
    staging = artifacts_root / ".kaggle-staging"
    published = artifacts_root / "kaggle"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    archive_path = _archive(staging)
    title = "RankLab Experiment Artifacts"
    (staging / "dataset-metadata.json").write_text(json.dumps({
        "title": title,
        "id": arguments.dataset,
        "licenses": [{"name": "other"}],
    }, indent=2) + "\n")
    shutil.rmtree(published, ignore_errors=True)
    os.replace(staging, published)
    archive_path = published / archive_path.name
    print(f"packaged {archive_path}")
    if arguments.no_upload:
        return
    if arguments.create:
        command = ["kaggle", "datasets", "create", "-p", str(published)]
        if arguments.public:
            command.append("-u")
    else:
        command = ["kaggle", "datasets", "version", "-p", str(published), "-m", arguments.message]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
