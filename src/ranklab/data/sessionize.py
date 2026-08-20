from __future__ import annotations

import pandas as pd


def add_ranking_contexts(
    interactions: pd.DataFrame,
    max_gap_minutes: int = 30,
    window_size: int = 20,
) -> pd.DataFrame:
    """Assign deterministic synthetic ranking contexts without claiming request IDs."""
    required = {"user_id", "timestamp_ms", "item_id"}
    if missing := required - set(interactions):
        raise ValueError(f"ranking context input is missing {sorted(missing)}")
    if max_gap_minutes <= 0 or window_size <= 0:
        raise ValueError("max_gap_minutes and window_size must be positive")
    work = interactions.sort_values(
        ["user_id", "timestamp_ms", "item_id"], kind="stable"
    ).copy()
    gap = work.groupby("user_id", sort=False)["timestamp_ms"].diff()
    new_session = gap.isna() | gap.gt(max_gap_minutes * 60_000)
    work["_session"] = new_session.groupby(work["user_id"], sort=False).cumsum() - 1
    work["session_position"] = work.groupby(
        ["user_id", "_session"], sort=False
    ).cumcount()
    work["_window"] = work["session_position"] // window_size
    work["context_id"] = (
        work["user_id"].astype(str)
        + ":"
        + work["_session"].astype(str)
        + ":"
        + work["_window"].astype(str)
    )
    return work.drop(columns=["_session", "_window"])
