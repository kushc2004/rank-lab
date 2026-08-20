from __future__ import annotations

import pandas as pd

def build_historical_features(interactions: pd.DataFrame, alpha: float = 20.0) -> pd.DataFrame:
    """Features at a row use only strictly preceding timestamp rows (never labels at t)."""
    ordered = interactions.sort_values(["timestamp_ms", "user_id", "item_id"], kind="stable").copy()
    # Aggregate each timestamp first. Subtracting the present block from each
    # cumulative sum makes every feature strictly earlier than timestamp t,
    # including for rows sharing exactly the same timestamp.
    ordered["_positive"] = ordered["long_view"].astype("int64")
    global_stats = ordered.groupby("timestamp_ms", sort=False).agg(_n=("item_id", "size"), _positive=("_positive", "sum"))
    global_stats["_prior_n"] = global_stats["_n"].cumsum() - global_stats["_n"]
    global_stats["_prior_positive"] = global_stats["_positive"].cumsum() - global_stats["_positive"]
    global_stats["_global_rate"] = global_stats["_prior_positive"].div(global_stats["_prior_n"]).fillna(0.0)
    ordered = ordered.join(global_stats[["_global_rate"]], on="timestamp_ms")

    def historical_counts(key: str, prefix: str) -> pd.DataFrame:
        stats = ordered.groupby(["timestamp_ms", key], sort=False).agg(
            _n=("item_id", "size"), _positive=("_positive", "sum")
        ).reset_index()
        stats["_prior_n"] = stats.groupby(key, sort=False)["_n"].cumsum() - stats["_n"]
        stats["_prior_positive"] = stats.groupby(key, sort=False)["_positive"].cumsum() - stats["_positive"]
        return stats.rename(columns={"_prior_n": f"hist_{prefix}_exposure", "_prior_positive": f"_hist_{prefix}_positive"})[
            ["timestamp_ms", key, f"hist_{prefix}_exposure", f"_hist_{prefix}_positive"]
        ]

    ordered = ordered.merge(historical_counts("item_id", "item"), on=["timestamp_ms", "item_id"], how="left", validate="many_to_one")
    ordered = ordered.merge(historical_counts("user_id", "user"), on=["timestamp_ms", "user_id"], how="left", validate="many_to_one")
    ordered["hist_item_reward_rate"] = (ordered["_hist_item_positive"] + alpha * ordered["_global_rate"]) / (ordered["hist_item_exposure"] + alpha)
    ordered["hist_user_long_view_rate"] = (ordered["_hist_user_positive"] + alpha * ordered["_global_rate"]) / (ordered["hist_user_exposure"] + alpha)
    ordered["feature_cutoff_ms"] = ordered["timestamp_ms"] - 1
    return ordered.drop(columns=["_positive", "_global_rate", "_hist_item_positive", "_hist_user_positive"])
