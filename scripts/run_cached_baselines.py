#!/usr/bin/env python3
"""Run only baseline stages whose verified artifacts are not already cached.

The cache deliberately contains derived experiment artifacts only.  Raw
KuaiRand files remain an external Kaggle input (or a local ignored directory).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ranklab.utils.config import kuairand_config


ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 2
STAGES = {
    "audit": (
        "scripts/audit_kuairand.py",
        (
            "data/manifests/train.parquet",
            "data/manifests/validation.parquet",
            "data/manifests/standard_test.parquet",
            "data/manifests/randomized_test.parquet",
            "data/manifests/split_manifest.parquet",
            "outputs/reports/kuairand_data_audit.json",
            "outputs/reports/kuairand_data_audit.md",
            "outputs/reports/feature_leakage_audit.csv",
        ),
    ),
    "features": (
        "scripts/build_features.py",
        ("data/features/train_historical.parquet", "data/features/feature_manifest.json"),
    ),
    "popularity": (
        "scripts/train_popularity.py",
        (
            "outputs/models/popularity.pkl",
            "outputs/metrics/popularity.json",
            "outputs/predictions/popularity_standard.parquet",
            "outputs/predictions/popularity_standard_per_group.parquet",
            "outputs/predictions/popularity_randomized.parquet",
            "outputs/predictions/popularity_randomized_per_group.parquet",
        ),
    ),
    "bpr": (
        "scripts/train_bpr.py",
        (
            "outputs/models/bpr.pkl",
            "outputs/metrics/bpr.json",
            "outputs/predictions/bpr_standard.parquet",
            "outputs/predictions/bpr_standard_per_group.parquet",
            "outputs/predictions/bpr_randomized.parquet",
            "outputs/predictions/bpr_randomized_per_group.parquet",
        ),
    ),
    "exposure_gap": (
        "scripts/evaluate_exposure_gap.py",
        ("outputs/reports/initial_exposure_gap.md",),
    ),
}


def _source_fingerprint(raw_dir: Path) -> dict[str, object]:
    """Identify cache compatibility without copying or hashing raw data."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in raw_dir={raw_dir}")
    return {
        "cache_version": CACHE_VERSION,
        "raw_files": [{"name": path.name, "bytes": path.stat().st_size} for path in csv_files],
        "implementation_sha256": _implementation_hash(),
    }


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    tracked = [ROOT / "configs", ROOT / "src/ranklab"]
    files = [path for directory in tracked for path in directory.rglob("*") if path.is_file()]
    files.extend(ROOT / script for script, _ in STAGES.values())
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _is_complete(stage: str) -> bool:
    return all((ROOT / relative).is_file() for relative in STAGES[stage][1])


def main() -> None:
    args = sys.argv[1:]
    force = "--force" in args
    bootstrap = "--bootstrap-existing" in args
    args = [arg for arg in args if arg not in {"--force", "--bootstrap-existing"}]
    config = kuairand_config(args)
    raw_dir = Path(config["raw_dir"]).expanduser().resolve()
    expected = _source_fingerprint(raw_dir)
    manifest_path = ROOT / "outputs/run_cache_manifest.json"
    existing = _load_manifest(manifest_path)
    compatible = existing is not None and existing.get("fingerprint") == expected
    completed = set(existing.get("completed", [])) if compatible and not force else set()
    if bootstrap:
        completed = {stage for stage in STAGES if _is_complete(stage)}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"fingerprint": expected, "completed": sorted(completed)}, indent=2) + "\n")
        print(f"registered existing complete stages: {', '.join(sorted(completed)) or 'none'}")
        return

    for stage, (script, _) in STAGES.items():
        if stage in completed and _is_complete(stage):
            print(f"cache hit: {stage}")
            continue
        print(f"running: {stage}")
        subprocess.run([sys.executable, script, *args], cwd=ROOT, check=True)
        if not _is_complete(stage):
            raise RuntimeError(f"{stage} finished without all expected artifacts")
        completed.add(stage)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"fingerprint": expected, "completed": sorted(completed)}, indent=2) + "\n")

    print(f"cache manifest: {manifest_path}")


if __name__ == "__main__":
    main()
