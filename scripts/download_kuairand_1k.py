#!/usr/bin/env python3
"""Download and verify the official KuaiRand-1K archive from Zenodo.

The archive name and checksum below are published by the official KuaiRand
repository.  The script never renames source files and extracts only the
official archive layout.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile
from urllib.request import urlretrieve

from ranklab.data.kuairand_1k import (
    OFFICIAL_1K_ARCHIVE,
    OFFICIAL_1K_MD5,
    OFFICIAL_1K_URL,
    raw_paths,
)


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - published archive-integrity checksum
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"archive contains an unsafe path: {member.name}")
        bundle.extractall(destination, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / OFFICIAL_1K_ARCHIVE
    if not archive.is_file():
        print(f"[download] {OFFICIAL_1K_URL}", flush=True)
        urlretrieve(OFFICIAL_1K_URL, archive)
    observed = md5(archive)
    if observed != OFFICIAL_1K_MD5:
        raise SystemExit(
            f"checksum mismatch for {archive}: expected {OFFICIAL_1K_MD5}, got {observed}"
        )
    print(f"[verified] {archive} md5={observed}", flush=True)
    if args.extract:
        safe_extract(archive, args.destination)
        # The official archive root is KuaiRand-1K/data. Validate that exact layout.
        root = args.destination / "KuaiRand-1K"
        raw_paths(root)
        print(f"[extracted] {root}", flush=True)


if __name__ == "__main__":
    main()
