"""Contracts and streaming inspection for the official KuaiRand-1K release.

This module deliberately does not route 1K rows into the KuaiRand-Pure
experiment.  1K is a separate, substantially larger corpus and must be
audited and trained as a separate experiment before it is used for retrieval
scale claims.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

from .schemas import INTERACTION_REQUIRED


OFFICIAL_1K_ARCHIVE = "KuaiRand-1K.tar.gz"
OFFICIAL_1K_MD5 = "6b0b9c8222d67fcd4c676218edca3f1f"
OFFICIAL_1K_URL = (
    "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz"
)
OFFICIAL_1K_FILES = {
    "random": "log_random_4_22_to_5_08_1k.csv",
    "standard_early": "log_standard_4_08_to_4_21_1k.csv",
    "standard_late": "log_standard_4_22_to_5_08_1k.csv",
    "users": "user_features_1k.csv",
    "items_basic": "video_features_basic_1k.csv",
    "items_statistic": "video_features_statistic_1k.csv",
}


def _official_data_dir(raw_dir: str | Path) -> Path:
    """Accept either the official extraction root or its official ``data`` dir."""
    root = Path(raw_dir).expanduser()
    if (root / "data").is_dir():
        return root / "data"
    return root


def raw_paths(raw_dir: str | Path) -> dict[str, Path]:
    data_dir = _official_data_dir(raw_dir)
    paths = {name: data_dir / filename for name, filename in OFFICIAL_1K_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "KuaiRand-1K official files are absent. Expected the extracted official "
            f"structure under {data_dir}: " + ", ".join(missing)
        )
    return paths


def interaction_paths(raw_dir: str | Path) -> dict[str, tuple[Path, int]]:
    paths = raw_paths(raw_dir)
    return {
        "standard_early": (paths["standard_early"], 0),
        "standard_late": (paths["standard_late"], 0),
        "random": (paths["random"], 1),
    }


def iter_interaction_chunks(
    raw_dir: str | Path, *, chunksize: int = 250_000
) -> Iterator[tuple[str, int, pd.DataFrame]]:
    """Yield validated raw interaction chunks without materialising 1K in RAM."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    for source_log, (path, expected_random) in interaction_paths(raw_dir).items():
        header = pd.read_csv(path, nrows=0)
        missing = sorted(set(INTERACTION_REQUIRED).difference(header.columns))
        if missing:
            raise ValueError(f"{path.name} lacks required official fields: {missing}")
        for chunk in pd.read_csv(path, chunksize=chunksize):
            random_values = pd.to_numeric(chunk["is_rand"], errors="raise")
            if not random_values.isin([0, 1]).all():
                raise ValueError(f"{path.name} has non-binary is_rand values")
            if not random_values.eq(expected_random).all():
                raise ValueError(
                    f"{path.name} contradicts its expected random/standard source"
                )
            yield source_log, expected_random, chunk


def streaming_interaction_audit(
    raw_dir: str | Path, *, chunksize: int = 250_000
) -> list[dict[str, object]]:
    """Return exact per-file row/user/item/date counts with bounded row memory."""
    rows: dict[str, dict[str, object]] = defaultdict(dict)
    users: dict[str, set[int]] = defaultdict(set)
    items: dict[str, set[int]] = defaultdict(set)
    for source_log, is_random, chunk in iter_interaction_chunks(raw_dir, chunksize=chunksize):
        required = ("user_id", "video_id", "date", "time_ms")
        numeric = {
            column: pd.to_numeric(chunk[column], errors="raise") for column in required
        }
        row = rows.setdefault(source_log, {
            "source_log": source_log,
            "is_random": int(is_random),
            "rows": 0,
            "min_date": None,
            "max_date": None,
            "min_timestamp_ms": None,
            "max_timestamp_ms": None,
        })
        row["rows"] = int(row["rows"]) + len(chunk)
        users[source_log].update(numeric["user_id"].astype("int64").tolist())
        items[source_log].update(numeric["video_id"].astype("int64").tolist())
        for key, values in (("date", numeric["date"]), ("timestamp_ms", numeric["time_ms"])):
            minimum, maximum = int(values.min()), int(values.max())
            min_key, max_key = f"min_{key}", f"max_{key}"
            row[min_key] = minimum if row[min_key] is None else min(int(row[min_key]), minimum)
            row[max_key] = maximum if row[max_key] is None else max(int(row[max_key]), maximum)
    result = []
    for source_log in ("standard_early", "standard_late", "random"):
        row = rows[source_log]
        row["users"] = len(users[source_log])
        row["items"] = len(items[source_log])
        result.append(row)
    return result


def discovered_file_summary(raw_dir: str | Path) -> list[dict[str, object]]:
    """Headers and byte sizes for all official 1K files (no full CSV read)."""
    rows = []
    for role, path in raw_paths(raw_dir).items():
        header = pd.read_csv(path, nrows=0)
        rows.append({
            "role": role,
            "file": path.name,
            "bytes": path.stat().st_size,
            "columns": list(header.columns),
            "sample_dtypes": {column: str(dtype) for column, dtype in header.dtypes.items()},
        })
    return rows
