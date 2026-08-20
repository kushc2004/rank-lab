from __future__ import annotations

from pathlib import Path

import numpy as np

from ranklab.data.open_bandit import OpenBanditFeedback, load_open_bandit_feedback


def load_behavior_and_ground_truth(
    *,
    data_path: str | Path | None,
    behavior_policy: str,
    campaign: str,
) -> tuple[OpenBanditFeedback, OpenBanditFeedback]:
    """Load propensity feedback plus a disjoint random-policy ground-truth log."""
    behavior = load_open_bandit_feedback(
        data_path=data_path,
        behavior_policy=behavior_policy,
        campaign=campaign,
    )
    random = load_open_bandit_feedback(
        data_path=data_path,
        behavior_policy="random",
        campaign=campaign,
    )
    if behavior.n_actions != random.n_actions:
        raise ValueError("behavior and random Open Bandit logs disagree on n_actions")
    return behavior, random


def uniform_action_distribution(n_rounds: int, n_actions: int) -> np.ndarray:
    if n_rounds < 1 or n_actions < 1:
        raise ValueError("uniform policy requires positive round and action counts")
    return np.full((n_rounds, n_actions), 1.0 / n_actions, dtype=np.float64)
