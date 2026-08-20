from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = (
    "retrieval_score", "retrieval_rank", "hist_item_exposure",
    "hist_item_reward_rate", "hist_user_exposure", "hist_user_long_view_rate",
    "hour", "day_of_week", "tab", "session_position",
    "video_duration", "register_days", "follow_user_num", "fans_user_num",
    "friend_user_num",
)


def training_aggregates(train: pd.DataFrame, alpha: float = 20.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rate = float(train["long_view"].mean())
    item = train.groupby("item_id", as_index=False).agg(
        hist_item_exposure=("long_view", "size"), _item_positive=("long_view", "sum")
    )
    item["hist_item_reward_rate"] = (
        item["_item_positive"] + alpha * global_rate
    ) / (item["hist_item_exposure"] + alpha)
    user = train.groupby("user_id", as_index=False).agg(
        hist_user_exposure=("long_view", "size"), _user_positive=("long_view", "sum")
    )
    user["hist_user_long_view_rate"] = (
        user["_user_positive"] + alpha * global_rate
    ) / (user["hist_user_exposure"] + alpha)
    return item.drop(columns="_item_positive"), user.drop(columns="_user_positive")


def point_in_time_aggregates(rows: pd.DataFrame, history: pd.DataFrame, alpha: float = 20.0) -> pd.DataFrame:
    """Attach cumulative statistics from events with timestamp strictly below each row."""
    result = rows.copy()
    result["_row_id"] = np.arange(len(result))
    events = history[["timestamp_ms", "user_id", "item_id", "long_view"]].copy()
    events["long_view"] = events["long_view"].astype(int)

    global_stats = events.groupby("timestamp_ms", as_index=False).agg(_n=("long_view", "size"), _p=("long_view", "sum")).sort_values("timestamp_ms")
    global_stats["_global_n"] = global_stats["_n"].cumsum()
    global_stats["_global_p"] = global_stats["_p"].cumsum()
    result = pd.merge_asof(
        result.sort_values("timestamp_ms"), global_stats[["timestamp_ms", "_global_n", "_global_p"]],
        on="timestamp_ms", direction="backward", allow_exact_matches=False,
    )

    for key, prefix in (("item_id", "item"), ("user_id", "user")):
        stats = events.groupby(["timestamp_ms", key], as_index=False).agg(_n=("long_view", "size"), _p=("long_view", "sum"))
        stats = stats.sort_values([key, "timestamp_ms"], kind="stable")
        stats[f"hist_{prefix}_exposure"] = stats.groupby(key, sort=False)["_n"].cumsum()
        stats[f"_hist_{prefix}_positive"] = stats.groupby(key, sort=False)["_p"].cumsum()
        result = pd.merge_asof(
            result.sort_values(["timestamp_ms", key], kind="stable"),
            stats[["timestamp_ms", key, f"hist_{prefix}_exposure", f"_hist_{prefix}_positive"]].sort_values(["timestamp_ms", key], kind="stable"),
            on="timestamp_ms", by=key, direction="backward", allow_exact_matches=False,
        )
    result = result.sort_values("_row_id").drop(columns="_row_id")
    global_rate = result["_global_p"].fillna(0) / result["_global_n"].replace(0, np.nan)
    global_rate = global_rate.fillna(0)
    for prefix in ("item", "user"):
        result[f"hist_{prefix}_exposure"] = result[f"hist_{prefix}_exposure"].fillna(0)
        positive = result.pop(f"_hist_{prefix}_positive").fillna(0)
        rate_name = "hist_item_reward_rate" if prefix == "item" else "hist_user_long_view_rate"
        result[rate_name] = (positive + alpha * global_rate) / (result[f"hist_{prefix}_exposure"] + alpha)
    return result.drop(columns=["_global_n", "_global_p"])


def build_ranker_features(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    point_in_time: bool = False,
    use_side_features: bool = False,
) -> pd.DataFrame:
    """Join leakage-safe historical and exposure-time features.

    Snapshot side-table fields are excluded by default because KuaiRand-Pure
    does not publish an as-of timestamp for them.  ``use_side_features`` is
    reserved for a clearly labelled relaxed sensitivity analysis.
    """
    if point_in_time:
        result = point_in_time_aggregates(rows, train)
    else:
        item_agg, user_agg = training_aggregates(train)
        result = rows.merge(item_agg, on="item_id", how="left", validate="many_to_one")
        result = result.merge(user_agg, on="user_id", how="left", validate="many_to_one")
    if use_side_features:
        user_columns = ["user_id", "register_days", "follow_user_num", "fans_user_num", "friend_user_num"]
        item_columns = ["item_id", "video_duration"]
        user_side = users[[c for c in user_columns if c in users]]
        item_side = items.rename(columns={"video_id": "item_id"})
        item_side = item_side[[c for c in item_columns if c in item_side]]
        result = result.merge(user_side, on="user_id", how="left", validate="many_to_one")
        result = result.merge(item_side, on="item_id", how="left", validate="many_to_one")
    timestamp = pd.to_datetime(result["timestamp_ms"], unit="ms", utc=True)
    result["hour"] = timestamp.dt.hour
    result["day_of_week"] = timestamp.dt.dayofweek
    for column in FEATURE_COLUMNS:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
