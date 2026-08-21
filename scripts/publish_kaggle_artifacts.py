#!/usr/bin/env python3
"""Package derived RankLab artifacts and optionally publish them with Kaggle CLI."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "kushchaudhari/ranklab-baseline-artifacts"
INCLUDE = ("data/manifests", "data/features", "data/processed", "data/indices", "outputs")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_complete_pipeline() -> dict:
    """Reject a cache from a failed or partial full-pipeline execution."""
    state_path = ROOT / "outputs/full_pipeline_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing pipeline state: {state_path}")
    state = json.loads(state_path.read_text())
    stages = state.get("stages", {})
    expected = ("baselines", "two_tower", "candidates", "rankers", "evaluation", "analysis", "ope", "scale", "serving", "report")
    incomplete = {
        name: stages.get(name, {}).get("status", "missing")
        for name in expected
        if stages.get(name, {}).get("status") not in {"complete", "skipped"}
    }
    if incomplete:
        raise RuntimeError(f"Refusing to publish an incomplete pipeline cache: {incomplete}")
    missing_artifacts = []
    for name in expected:
        for relative in stages[name].get("artifacts", []):
            if not isinstance(relative, str):
                missing_artifacts.append(f"{name}:{relative}")
                continue
            path = ROOT / relative
            try:
                path.resolve().relative_to(ROOT.resolve())
            except ValueError:
                missing_artifacts.append(f"{name}:{relative}")
                continue
            if not path.is_file():
                missing_artifacts.append(f"{name}:{relative}")
    if missing_artifacts:
        raise RuntimeError(
            "Refusing to publish a cache with missing recorded artifacts: "
            f"{missing_artifacts[:3]}"
        )
    return state


def _write_manifest() -> Path:
    """Record every source file that must be present in the reusable archive."""
    files = []
    for relative in INCLUDE:
        directory = ROOT / relative
        if directory.is_dir():
            for path in sorted(item for item in directory.rglob("*") if item.is_file() and item.name != ".DS_Store"):
                files.append({
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                })
    manifest_path = ROOT / "outputs/kaggle_artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"files": files}, indent=2) + "\n")
    return manifest_path


def _archive(staging: Path) -> Path:
    archive_path = staging / "ranklab_artifacts.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
        for relative in INCLUDE:
            directory = ROOT / relative
            if directory.is_dir():
                for file_path in sorted(path for path in directory.rglob("*") if path.is_file() and path.name != ".DS_Store"):
                    archive.add(file_path, arcname=file_path.relative_to(ROOT))
    return archive_path


def _verify_archive(archive_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    expected = {record["path"]: record for record in manifest["files"]}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        missing = sorted(set(expected) - set(members))
        if missing:
            raise RuntimeError(f"Archive is missing expected files: {missing[:3]}")
        raw = [name for name in members if name.startswith("data/raw/")]
        if raw:
            raise RuntimeError(f"Archive unexpectedly contains raw data: {raw[:3]}")
        for path, record in expected.items():
            source = archive.extractfile(members[path])
            if source is None:
                raise RuntimeError(f"Could not read archived artifact: {path}")
            digest = hashlib.sha256(source.read()).hexdigest()
            if digest != record["sha256"]:
                raise RuntimeError(f"Archive checksum mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--create", action="store_true", help="Create the dataset; otherwise publish a new version.")
    parser.add_argument("--public", action="store_true", help="Only applies with --create. Default is private.")
    parser.add_argument("--message", default="Refresh RankLab derived experiment artifacts")
    parser.add_argument("--no-upload", action="store_true", help="Package only, for a Kaggle notebook output.")
    parser.add_argument("--require-complete", action="store_true", help="Require every full-pipeline stage to be complete or truthfully skipped.")
    arguments = parser.parse_args()

    if arguments.require_complete:
        _require_complete_pipeline()
    manifest_path = _write_manifest()
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
    _verify_archive(archive_path, manifest_path)
    print(f"packaged {archive_path}")
    print(f"verified {manifest_path} ({archive_path.stat().st_size} bytes)")
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
