from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def rerank_context(frame: pd.DataFrame, preference: dict[str, float], relevance_weight: float, k: int) -> pd.DataFrame:
    """Greedy relevance/calibration reranking; input candidates are never mutated."""
    remaining = frame.sort_values(["score", "item_id"], ascending=[False, True], kind="stable").copy()
    # The previous implementation iterated each candidate with ``iterrows`` at
    # every greedy step.  A frontier has many contexts and weights, so that
    # turned a small 10-item rerank into millions of Python/Pandas operations.
    # The calibration cost depends only on the candidate category; calculate
    # it once per category at each step and map it back to candidates.
    categories = remaining["category"].fillna("unknown").astype(str).to_numpy()
    universe = sorted(set(categories) | set(preference))
    pref = np.asarray([preference.get(value, 0.0) for value in universe], dtype=float)
    if pref.sum() == 0:
        pref = np.ones(len(universe), dtype=float)
    pref /= pref.sum()
    category_to_code = {category: code for code, category in enumerate(universe)}
    category_codes = np.fromiter(
        (category_to_code[category] for category in categories), dtype=np.intp, count=len(categories)
    )
    scores = remaining["score"].to_numpy(float)
    low, high = scores.min(initial=0), scores.max(initial=0)
    normalized = (scores - low) / max(high - low, 1e-12)
    selected: list[int] = []
    selected_mask = np.zeros(len(remaining), dtype=bool)
    selected_counts = np.zeros(len(universe), dtype=float)
    category_increment = np.eye(len(universe), dtype=float)
    for selected_count in range(min(k, len(remaining))):
        distributions = (selected_counts + category_increment) / (selected_count + 1)
        category_costs = np.abs(distributions - pref).sum(axis=1) / 2
        utility = relevance_weight * normalized - (1 - relevance_weight) * category_costs[category_codes]
        utility[selected_mask] = -np.inf
        # np.argmax retains the earliest position on ties, matching the stable
        # candidate order used by the former strict-greater-than comparison.
        best_position = int(np.argmax(utility))
        selected.append(best_position)
        selected_mask[best_position] = True
        selected_counts[category_codes[best_position]] += 1
    # Ranking metrics require the complete candidate list: rerank only the head,
    # then append the untouched tail in its original score order.
    selected_set = set(selected)
    tail = [position for position in range(len(remaining)) if position not in selected_set]
    result = remaining.iloc[selected + tail].copy()
    result["rerank_position"] = np.arange(1, len(result) + 1)
    result["relevance_weight"] = relevance_weight
    return result


def rerank_frontier(
    candidates: pd.DataFrame,
    profiles: dict[int, dict[str, float]],
    weights: tuple[float, ...],
    k: int = 10,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    records = []
    groups = list(candidates.groupby(["context_id", "user_id"], sort=False))
    for weight in weights:
        if progress:
            progress(f"reranking weight={weight:g} (0/{len(groups)} contexts)")
        for count, ((_, user_id), frame) in enumerate(groups, start=1):
            records.append(rerank_context(frame, profiles.get(int(user_id), {}), weight, k))
            if progress and (count % 500 == 0 or count == len(groups)):
                progress(f"reranking weight={weight:g} ({count}/{len(groups)} contexts)")
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()
