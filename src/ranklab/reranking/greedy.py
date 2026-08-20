from __future__ import annotations

import numpy as np
import pandas as pd


def _distribution(categories: list[str], universe: list[str]) -> np.ndarray:
    counts = pd.Series(categories).value_counts()
    return np.asarray([counts.get(value, 0) for value in universe], dtype=float)


def rerank_context(frame: pd.DataFrame, preference: dict[str, float], relevance_weight: float, k: int) -> pd.DataFrame:
    """Greedy relevance/calibration reranking; input candidates are never mutated."""
    remaining = frame.sort_values(["score", "item_id"], ascending=[False, True], kind="stable").copy()
    selected: list[int] = []
    categories: list[str] = []
    universe = sorted(set(remaining["category"].fillna("unknown")) | set(preference))
    pref = np.asarray([preference.get(value, 0.0) for value in universe], dtype=float)
    if pref.sum() == 0:
        pref = np.ones(len(universe), dtype=float)
    pref /= pref.sum()
    scores = remaining["score"].to_numpy(float)
    low, high = scores.min(initial=0), scores.max(initial=0)
    normalized = (scores - low) / max(high - low, 1e-12)
    for _ in range(min(k, len(remaining))):
        best_position, best_utility = None, -np.inf
        for position, (_, row) in enumerate(remaining.iterrows()):
            if position in selected:
                continue
            candidate_categories = categories + [str(row.get("category", "unknown"))]
            distribution = _distribution(candidate_categories, universe)
            distribution /= max(distribution.sum(), 1)
            calibration_cost = float(np.abs(distribution - pref).sum() / 2)
            utility = relevance_weight * normalized[position] - (1 - relevance_weight) * calibration_cost
            if utility > best_utility:
                best_position, best_utility = position, utility
        assert best_position is not None
        selected.append(best_position)
        categories.append(str(remaining.iloc[best_position].get("category", "unknown")))
    # Ranking metrics require the complete candidate list: rerank only the head,
    # then append the untouched tail in its original score order.
    selected_set = set(selected)
    tail = [position for position in range(len(remaining)) if position not in selected_set]
    result = remaining.iloc[selected + tail].copy()
    result["rerank_position"] = np.arange(1, len(result) + 1)
    result["relevance_weight"] = relevance_weight
    return result


def rerank_frontier(candidates: pd.DataFrame, profiles: dict[int, dict[str, float]], weights: tuple[float, ...], k: int = 10) -> pd.DataFrame:
    records = []
    for weight in weights:
        for (_, user_id), frame in candidates.groupby(["context_id", "user_id"], sort=False):
            records.append(rerank_context(frame, profiles.get(int(user_id), {}), weight, k))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
