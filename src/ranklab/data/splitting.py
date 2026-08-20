from __future__ import annotations

from pathlib import Path
import pandas as pd

SPLITS = ("train", "validation", "standard_test", "randomized_test")

def assign_splits(interactions: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = interactions.copy()
    result["split"] = ""
    standard = result["is_random"].eq(0)
    random = result["is_random"].eq(1)
    result.loc[standard & result.date.le(int(config["train_end"])), "split"] = "train"
    result.loc[standard & result.date.between(int(config["validation_start"]), int(config["validation_end"])), "split"] = "validation"
    result.loc[standard & result.date.between(int(config["standard_test_start"]), int(config["standard_test_end"])), "split"] = "standard_test"
    result.loc[random & result.date.between(int(config["randomized_test_start"]), int(config["randomized_test_end"])), "split"] = "randomized_test"
    if (result["split"] == "").any():
        raise ValueError("Some interactions are outside configured temporal split bounds")
    return result

def write_manifests(interactions: pd.DataFrame, config: dict) -> dict[str, int]:
    manifest_dir = Path(config["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        frame = interactions.loc[interactions["split"].eq(split)].copy()
        frame.to_parquet(manifest_dir / f"{split}.parquet", index=False)
        counts[split] = len(frame)
    interactions[["user_id", "item_id", "timestamp_ms", "date", "is_random", "source_log", "split"]].to_parquet(manifest_dir / "split_manifest.parquet", index=False)
    return counts

def validate_splits(interactions: pd.DataFrame) -> None:
    train = interactions.loc[interactions["split"].eq("train")]
    later_standard = interactions.loc[interactions["split"].isin(["validation", "standard_test"])]
    randomized = interactions.loc[interactions["split"].eq("randomized_test")]
    if train["is_random"].any() or not randomized["is_random"].all():
        raise AssertionError("Randomized interactions entered training or standard logs entered randomized test")
    # KuaiRand-Pure's official calendar `date` is the split authority.  Its
    # millisecond clock has a small timezone-boundary overlap between adjacent
    # dates (for example, late 20220421 and early 20220422), so comparing raw
    # epoch milliseconds across those official day partitions is invalid.
    # Milliseconds remain the event order within each split/feature history.
    if not train.empty and not later_standard.empty and train["date"].max() >= later_standard["date"].min():
        raise AssertionError("Train dates overlap later standard splits")
    if set(interactions["split"]) != set(SPLITS):
        raise AssertionError("Not all required splits are represented")
