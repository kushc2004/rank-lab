from __future__ import annotations

import ast
import numpy as np
import pandas as pd


def primary_category(value: object) -> str:
    """Return a stable first category from KuaiRand's serialized tag field."""
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)) and parsed:
            return str(next(iter(parsed)))
    except (ValueError, SyntaxError):
        pass
    for separator in (",", "|", ";"):
        if separator in text:
            return text.split(separator, 1)[0].strip(" []'\"") or "unknown"
    return text.strip(" []'\"") or "unknown"


def jensen_shannon(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> float:
    left = np.asarray(left, dtype=float) + epsilon
    right = np.asarray(right, dtype=float) + epsilon
    left, right = left / left.sum(), right / right.sum()
    middle = (left + right) / 2
    return float((np.sum(left * np.log(left / middle)) + np.sum(right * np.log(right / middle))) / 2)


def user_preference_profiles(train: pd.DataFrame, item_categories: pd.DataFrame) -> dict[int, dict[str, float]]:
    columns = item_categories.rename(columns={"video_id": "item_id"})[["item_id", "category"]]
    positive = train.loc[train["long_view"].eq(1), ["user_id", "item_id"]].merge(columns, on="item_id", how="left")
    counts = positive.assign(category=positive["category"].fillna("unknown")).groupby(["user_id", "category"]).size()
    profiles: dict[int, dict[str, float]] = {}
    for user_id, values in counts.groupby(level=0):
        series = values.droplevel(0).astype(float)
        profiles[int(user_id)] = (series / series.sum()).to_dict()
    return profiles


def calibration_report(
    recommendations: pd.DataFrame,
    profiles: dict[int, dict[str, float]],
    k: int = 10,
) -> dict[str, float]:
    top = recommendations.sort_values(["context_id", "score"], ascending=[True, False], kind="stable").groupby("context_id", sort=False).head(k)
    if top.empty or not profiles:
        return {"js_divergence_mean": float("nan"), "contexts": 0, "k": int(k)}

    # Expand fixed user profiles once, then join them to recommendation
    # category probabilities.  This replaces a Python loop over every ranking
    # context while retaining the original union-of-categories JS definition.
    profile_rows = [
        {"user_id": int(user_id), "category": str(category), "preference": float(value)}
        for user_id, preferences in profiles.items()
        for category, value in preferences.items()
    ]
    if not profile_rows:
        return {"js_divergence_mean": float("nan"), "contexts": 0, "k": int(k)}
    profile_frame = pd.DataFrame(profile_rows)
    work = top[["context_id", "user_id", "category"]].copy()
    work["category"] = work["category"].fillna("unknown").astype(str)
    context_users = work[["context_id", "user_id"]].drop_duplicates()
    profile_contexts = context_users.merge(profile_frame, on="user_id", how="inner", validate="many_to_many")
    if profile_contexts.empty:
        return {"js_divergence_mean": float("nan"), "contexts": 0, "k": int(k)}
    eligible = profile_contexts[["context_id", "user_id"]].drop_duplicates()
    recommendation_counts = (
        work.merge(eligible, on=["context_id", "user_id"], how="inner", validate="many_to_one")
        .groupby(["context_id", "user_id", "category"], sort=False)
        .size()
        .rename("recommendation_count")
        .reset_index()
    )
    recommendation_counts["recommendation"] = recommendation_counts["recommendation_count"] / recommendation_counts.groupby(
        ["context_id", "user_id"], sort=False
    )["recommendation_count"].transform("sum")
    union = profile_contexts.merge(
        recommendation_counts[["context_id", "user_id", "category", "recommendation"]],
        on=["context_id", "user_id", "category"], how="outer", validate="one_to_one",
    )
    union["preference"] = union["preference"].fillna(0.0)
    union["recommendation"] = union["recommendation"].fillna(0.0)
    group_columns = ["context_id", "user_id"]
    epsilon = 1e-12
    union["_left"] = union["preference"] + epsilon
    union["_right"] = union["recommendation"] + epsilon
    union["_left"] /= union.groupby(group_columns, sort=False)["_left"].transform("sum")
    union["_right"] /= union.groupby(group_columns, sort=False)["_right"].transform("sum")
    middle = (union["_left"] + union["_right"]) / 2
    union["_js_term"] = (
        union["_left"] * np.log(union["_left"] / middle)
        + union["_right"] * np.log(union["_right"] / middle)
    ) / 2
    divergences = union.groupby(group_columns, sort=False)["_js_term"].sum()
    return {
        "js_divergence_mean": float(divergences.mean()) if not divergences.empty else float("nan"),
        "contexts": int(len(divergences)),
        "k": int(k),
    }
