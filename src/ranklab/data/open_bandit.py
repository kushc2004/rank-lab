from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class OpenBanditFeedback:
    """Validated propensity-bearing contextual-bandit feedback from OBP."""

    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    n_actions: int
    behavior_policy: str
    campaign: str

    @property
    def n_rounds(self) -> int:
        return int(len(self.reward))


def validate_open_bandit_feedback(
    feedback: Mapping[str, Any],
    *,
    behavior_policy: str,
    campaign: str,
) -> OpenBanditFeedback:
    required = ("context", "action", "reward", "pscore", "n_actions")
    missing = [name for name in required if name not in feedback]
    if missing:
        raise ValueError(f"Open Bandit feedback is missing fields: {missing}")

    context = np.asarray(feedback["context"], dtype=np.float32)
    action = np.asarray(feedback["action"], dtype=np.int64)
    reward = np.asarray(feedback["reward"], dtype=np.float64)
    pscore = np.asarray(feedback["pscore"], dtype=np.float64)
    n_actions = int(feedback["n_actions"])

    if context.ndim != 2:
        raise ValueError("Open Bandit context must be a two-dimensional matrix")
    if action.ndim != 1 or reward.ndim != 1 or pscore.ndim != 1:
        raise ValueError("Open Bandit action, reward, and pscore must be one-dimensional")
    if not (len(context) == len(action) == len(reward) == len(pscore)):
        raise ValueError("Open Bandit feedback arrays have inconsistent lengths")
    if len(action) == 0 or n_actions < 1:
        raise ValueError("Open Bandit feedback must contain rounds and at least one action")
    if np.any((action < 0) | (action >= n_actions)):
        raise ValueError("Open Bandit feedback contains an out-of-range action")
    if np.any(~np.isfinite(context)) or np.any(~np.isfinite(reward)):
        raise ValueError("Open Bandit context or reward contains non-finite values")
    if np.any(~np.isfinite(pscore)) or np.any((pscore <= 0) | (pscore > 1)):
        raise ValueError("Open Bandit propensities must be finite and in (0, 1]")

    return OpenBanditFeedback(
        context=context,
        action=action,
        reward=reward,
        pscore=pscore,
        n_actions=n_actions,
        behavior_policy=behavior_policy,
        campaign=campaign,
    )


def load_open_bandit_feedback(
    *,
    data_path: str | Path | None,
    behavior_policy: str,
    campaign: str,
) -> OpenBanditFeedback:
    """Load OBD through its official OBP adapter without fabricating propensities."""
    try:
        from obp.dataset import OpenBanditDataset
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Open Bandit support requires the optional 'ope' dependencies"
        ) from error

    kwargs: dict[str, object] = {
        "behavior_policy": behavior_policy,
        "campaign": campaign,
    }
    if data_path is not None:
        kwargs["data_path"] = str(Path(data_path).expanduser().resolve())
    dataset = OpenBanditDataset(**kwargs)
    return validate_open_bandit_feedback(
        dataset.obtain_batch_bandit_feedback(),
        behavior_policy=behavior_policy,
        campaign=campaign,
    )
