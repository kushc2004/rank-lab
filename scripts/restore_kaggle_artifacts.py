#!/usr/bin/env python3
"""Safely restore a RankLab derived-artifact archive into this checkout."""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATA_DIRECTORIES = {"manifests", "features", "processed", "indices"}


def _allowed(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not member.isfile():
        return False
    if path.parts[0] == "outputs":
        return True
    return len(path.parts) >= 2 and path.parts[0] == "data" and path.parts[1] in ALLOWED_DATA_DIRECTORIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_source", type=Path, help="A .tar.gz cache archive or an expanded Kaggle input directory")
    arguments = parser.parse_args()
    source = arguments.artifact_source
    if source.is_dir():
        restored = []
        for relative in (Path("data/manifests"), Path("data/features"), Path("data/processed"), Path("data/indices"), Path("outputs")):
            input_directory = source / relative
            if not input_directory.is_dir():
                continue
            for input_file in input_directory.rglob("*"):
                if input_file.is_file():
                    destination = ROOT / input_file.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(input_file, destination)
                    restored.append(destination)
        if not restored:
            raise ValueError(f"No supported derived artifacts found in {source}")
        print(f"restored {len(restored)} derived artifacts from expanded input {source}")
        return
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        rejected = [member.name for member in members if not _allowed(member)]
        if rejected:
            raise ValueError(f"Archive contains unsafe or unsupported paths: {rejected[:3]}")
        archive.extractall(ROOT, members=members, filter="data")
    print(f"restored derived artifacts from {source}")


if __name__ == "__main__":
    main()
