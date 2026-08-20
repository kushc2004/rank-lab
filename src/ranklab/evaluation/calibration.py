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
    divergences: list[float] = []
    for (_, user_id), frame in top.groupby(["context_id", "user_id"], sort=False):
        preference = profiles.get(int(user_id), {})
        rec = frame["category"].fillna("unknown").value_counts(normalize=True).to_dict()
        categories = sorted(set(preference) | set(rec))
        if categories and preference:
            divergences.append(jensen_shannon([preference.get(c, 0) for c in categories], [rec.get(c, 0) for c in categories]))
    return {
        "js_divergence_mean": float(np.mean(divergences)) if divergences else float("nan"),
        "contexts": len(divergences),
        "k": int(k),
    }
