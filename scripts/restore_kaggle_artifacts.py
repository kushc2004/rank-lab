#!/usr/bin/env python3
"""Safely restore a RankLab derived-artifact archive into this checkout."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATA_DIRECTORIES = {"manifests", "features"}


def _allowed(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not member.isfile():
        return False
    if path.parts[0] == "outputs":
        return True
    return len(path.parts) >= 2 and path.parts[0] == "data" and path.parts[1] in ALLOWED_DATA_DIRECTORIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args()
    with tarfile.open(arguments.archive, "r:gz") as archive:
        members = archive.getmembers()
        rejected = [member.name for member in members if not _allowed(member)]
        if rejected:
            raise ValueError(f"Archive contains unsafe or unsupported paths: {rejected[:3]}")
        archive.extractall(ROOT, members=members, filter="data")
    print(f"restored derived artifacts from {arguments.archive}")


if __name__ == "__main__":
    main()
