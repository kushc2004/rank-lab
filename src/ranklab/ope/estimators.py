from __future__ import annotations

import numpy as np


def _validate_target_policy(target: np.ndarray, n_rounds: int) -> None:
    if target.ndim != 2 or target.shape[0] != n_rounds or target.shape[1] < 1:
        raise ValueError("target_action_dist must have shape (n_rounds, n_actions)")
    if not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError("target_action_dist must contain finite non-negative probabilities")
    if not np.allclose(target.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("target_action_dist rows must sum to one")


def importance_weights(
    action: np.ndarray,
    behavior_propensity: np.ndarray,
    target_action_dist: np.ndarray,
    clip: float | None = None,
) -> np.ndarray:
    action = np.asarray(action, dtype=int)
    propensity = np.asarray(behavior_propensity, dtype=float)
    target = np.asarray(target_action_dist, dtype=float)
    if action.ndim != 1:
        raise ValueError("action must be one-dimensional")
    _validate_target_policy(target, len(action))
    if propensity.shape != action.shape:
        raise ValueError("behavior_propensity and action must have the same shape")
    if np.any((action < 0) | (action >= target.shape[1])):
        raise ValueError("observed action falls outside target_action_dist")
    if np.any(~np.isfinite(propensity)) or np.any((propensity <= 0) | (propensity > 1)):
        raise ValueError("behavior propensities must be finite and in (0, 1]")
    rows = np.arange(len(action))
    weights = target[rows, action] / propensity
    if clip is not None:
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be a finite positive number")
        weights = np.minimum(weights, float(clip))
    return weights


def importance_weight_diagnostics(
    raw_weights: np.ndarray,
    clipped_weights: np.ndarray | None = None,
) -> dict[str, float]:
    raw = np.asarray(raw_weights, dtype=float)
    used = raw if clipped_weights is None else np.asarray(clipped_weights, dtype=float)
    if raw.ndim != 1 or used.shape != raw.shape:
        raise ValueError("importance weights must be equal-length one-dimensional arrays")
    if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(used)) or np.any(raw < 0) or np.any(used < 0):
        raise ValueError("importance weights must be finite and non-negative")
    total = used.sum()
    return {
        "minimum": float(used.min()) if len(used) else 0.0,
        "mean": float(used.mean()) if len(used) else 0.0,
        "standard_deviation": float(used.std()) if len(used) else 0.0,
        "p95": float(np.percentile(used, 95)) if len(used) else 0.0,
        "p99": float(np.percentile(used, 99)) if len(used) else 0.0,
        "maximum": float(used.max()) if len(used) else 0.0,
        "effective_sample_size": float(total * total / max(np.square(used).sum(), 1e-12)),
        "clipped_fraction": float(np.mean(used < raw)) if len(used) else 0.0,
    }


def estimate_policy_value(
    reward: np.ndarray,
    action: np.ndarray,
    behavior_propensity: np.ndarray,
    target_action_dist: np.ndarray,
    reward_prediction: np.ndarray,
    clip: float | None = None,
) -> dict[str, float]:
    """DM/IPS/SNIPS/DR for discrete contextual bandit feedback.

    target_action_dist and reward_prediction have shape (n_rounds, n_actions).
    behavior_propensity is the logging probability of the observed action.
    """
    reward = np.asarray(reward, dtype=float)
    action = np.asarray(action, dtype=int)
    propensity = np.asarray(behavior_propensity, dtype=float)
    target = np.asarray(target_action_dist, dtype=float)
    prediction = np.asarray(reward_prediction, dtype=float)
    if reward.ndim != 1 or reward.shape != action.shape or reward.shape != propensity.shape:
        raise ValueError("OPE arrays have inconsistent shapes")
    if len(reward) == 0:
        raise ValueError("cannot estimate policy value from zero rounds")
    _validate_target_policy(target, len(reward))
    if target.shape != prediction.shape:
        raise ValueError("reward_prediction must match target_action_dist")
    if not np.all(np.isfinite(reward)) or not np.all(np.isfinite(prediction)):
        raise ValueError("reward or reward_prediction contains non-finite values")
    rows = np.arange(len(reward))
    weights = importance_weights(action, propensity, target, clip)
    dm_per_round = np.sum(target * prediction, axis=1)
    residual = reward - prediction[rows, action]
    ips = np.mean(weights * reward)
    snips = np.sum(weights * reward) / max(np.sum(weights), 1e-12)
    dr = np.mean(dm_per_round + weights * residual)
    total = weights.sum()
    return {
        "dm": float(dm_per_round.mean()), "ips": float(ips), "snips": float(snips), "dr": float(dr),
        "effective_sample_size": float(total * total / max(np.square(weights).sum(), 1e-12)),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()) if len(weights) else 0.0,
        "clip": None if clip is None else float(clip), "n_rounds": int(len(reward)),
    }


def bootstrap_policy_values(
    reward: np.ndarray,
    action: np.ndarray,
    behavior_propensity: np.ndarray,
    target_action_dist: np.ndarray,
    reward_prediction: np.ndarray,
    clip: float | None,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Round-level nonparametric bootstrap intervals for all OPE estimators."""
    if samples < 1:
        return {}
    reward = np.asarray(reward)
    action = np.asarray(action)
    behavior_propensity = np.asarray(behavior_propensity)
    target_action_dist = np.asarray(target_action_dist)
    reward_prediction = np.asarray(reward_prediction)
    if len(reward) == 0:
        raise ValueError("cannot bootstrap policy value from zero rounds")
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ("dm", "ips", "snips", "dr")}
    for _ in range(samples):
        rows = rng.integers(0, len(reward), size=len(reward))
        estimate = estimate_policy_value(
            reward[rows], action[rows], behavior_propensity[rows],
            target_action_dist[rows], reward_prediction[rows], clip,
        )
        for name in draws:
            draws[name].append(estimate[name])
    return {
        name: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
            "bootstrap_standard_error": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for name, values in draws.items()
    }


def bootstrap_mean(values: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty array")
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("bootstrap values must be a finite one-dimensional array")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        values[rng.integers(0, len(values), size=len(values))].mean()
        for _ in range(samples)
    ])
    return {
        "estimate": float(values.mean()),
        "lower_95": float(np.percentile(draws, 2.5)),
        "upper_95": float(np.percentile(draws, 97.5)),
        "bootstrap_standard_error": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0,
        "bootstrap_samples": int(samples),
    }
