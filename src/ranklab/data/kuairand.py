from __future__ import annotations

from pathlib import Path
import pandas as pd

from .schemas import INTERACTION_REQUIRED, INTERACTION_RENAME

OFFICIAL_FILES = {
    "random": "log_random_4_22_to_5_08_pure.csv",
    "standard_early": "log_standard_4_08_to_4_21_pure.csv",
    "standard_late": "log_standard_4_22_to_5_08_pure.csv",
    "users": "user_features_pure.csv",
    "items_basic": "video_features_basic_pure.csv",
    "items_statistic": "video_features_statistic_pure.csv",
}

def raw_paths(raw_dir: str | Path) -> dict[str, Path]:
    root = Path(raw_dir)
    paths = {name: root / filename for name, filename in OFFICIAL_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("KuaiRand-Pure official files are absent: " + ", ".join(missing))
    return paths

def _read_interaction(path: Path, source_log: str, expected_random: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(INTERACTION_REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} lacks required official fields: {missing}")
    frame = frame.rename(columns=INTERACTION_RENAME).copy()
    if not frame["is_random"].isin([0, 1]).all():
        raise ValueError(f"{path.name} has non-binary is_rand values")
    if not (frame["is_random"] == expected_random).all():
        raise ValueError(f"{path.name} contradicts its expected random/standard source")
    frame["source_log"] = source_log
    for col in ("user_id", "item_id", "timestamp_ms", "date", "tab", "long_view"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype("int64")
    return frame.sort_values(["timestamp_ms", "user_id", "item_id"], kind="stable").reset_index(drop=True)

def load_interactions(raw_dir: str | Path) -> pd.DataFrame:
    paths = raw_paths(raw_dir)
    return pd.concat([
        _read_interaction(paths["standard_early"], "standard_early", 0),
        _read_interaction(paths["standard_late"], "standard_late", 0),
        _read_interaction(paths["random"], "random", 1),
    ], ignore_index=True).sort_values(["timestamp_ms", "user_id", "item_id"], kind="stable").reset_index(drop=True)

def load_side_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    paths = raw_paths(raw_dir)
    return {name: pd.read_csv(paths[name]) for name in ("users", "items_basic", "items_statistic")}

def discovered_file_summary(raw_dir: str | Path) -> list[dict]:
    paths = raw_paths(raw_dir)
    rows = []
    for name, path in paths.items():
        header = pd.read_csv(path, nrows=0)
        count = sum(1 for _ in path.open("rb")) - 1
        rows.append({"role": name, "file": path.name, "rows": count, "columns": list(header.columns)})
    return rows
