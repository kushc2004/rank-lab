#!/usr/bin/env python3
"""Run propensity-valid OPE against Open Bandit Dataset on-policy ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ranklab.ope.obp_adapter import (
    load_behavior_and_ground_truth,
    uniform_action_distribution,
)
from ranklab.ope.estimators import (
    bootstrap_mean,
    bootstrap_policy_values,
    estimate_policy_value,
    importance_weight_diagnostics,
    importance_weights,
)


def _reward_features(context: np.ndarray, action: np.ndarray, n_actions: int) -> np.ndarray:
    action_one_hot = np.eye(n_actions, dtype=np.float32)[np.asarray(action, dtype=int)]
    return np.column_stack([np.asarray(context, dtype=np.float32), action_one_hot])


def _fit_reward_model(
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    n_actions: int,
    seed: int,
):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    matrix = _reward_features(context, action, n_actions)
    if len(np.unique(reward)) < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = LogisticRegression(max_iter=500, random_state=seed, n_jobs=1)
    model.fit(matrix, reward)
    return model


def _positive_probability(model, matrix: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) == 0:
        return np.zeros(len(matrix), dtype=float)
    return probabilities[:, int(positive[0])]


def _reward_predictions(model, context: np.ndarray, n_actions: int) -> np.ndarray:
    result = np.empty((len(context), n_actions), dtype=np.float64)
    for candidate_action in range(n_actions):
        actions = np.full(len(context), candidate_action, dtype=int)
        result[:, candidate_action] = _positive_probability(
            model, _reward_features(context, actions, n_actions)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--campaign", default="all", choices=("all", "men", "women"))
    parser.add_argument("--behavior-policy", default="bts", choices=("bts",))
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/ope.json"))
    parser.add_argument("--sample-fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--clip-values", type=float, nargs="*", default=[10.0, 50.0])
    parser.add_argument(
        "overrides", nargs="*",
        help="documented selectors such as data=open_bandit experiment=obd_ope",
    )
    args = parser.parse_args()
    unsupported = [
        value for value in args.overrides
        if value not in {"data=open_bandit", "experiment=obd_ope"}
    ]
    if unsupported:
        raise ValueError(f"unsupported OPE overrides: {unsupported}")
    try:
        import obp  # noqa: F401
        import sklearn  # noqa: F401
    except ModuleNotFoundError as error:
        raise SystemExit("Install OPE dependencies with: pip install -e '.[full,ope]'") from error

    if any(not 0 < value <= 1 for value in args.sample_fractions):
        raise ValueError("sample fractions must be in (0, 1]")
    if any(value <= 0 for value in args.clip_values):
        raise ValueError("clip values must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")

    behavior, random = load_behavior_and_ground_truth(
        data_path=args.data_path,
        behavior_policy=args.behavior_policy,
        campaign=args.campaign,
    )
    context = behavior.context
    action = behavior.action
    reward = behavior.reward
    propensity = behavior.pscore
    n_actions = behavior.n_actions
    random_reward = random.reward
    if len(reward) < 2:
        raise ValueError("OPE benchmark needs at least two behavior-policy rounds")
    if not set(np.unique(reward)).issubset({0.0, 1.0}):
        raise ValueError("the configured reward model requires binary Open Bandit rewards")
    if not set(np.unique(random_reward)).issubset({0.0, 1.0}):
        raise ValueError("on-policy random-log rewards must be binary")

    ground_truth = bootstrap_mean(
        random_reward, args.bootstrap_samples, seed=min(args.seeds)
    )
    experiments: list[dict] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(reward))
        midpoint = len(order) // 2
        fit_rows, evaluation_pool = order[:midpoint], order[midpoint:]
        reward_model = _fit_reward_model(
            context[fit_rows], action[fit_rows], reward[fit_rows], n_actions, seed
        )
        for fraction in sorted(set(args.sample_fractions)):
            count = max(1, int(round(len(evaluation_pool) * fraction)))
            rows = evaluation_pool[:count]
            prediction = _reward_predictions(reward_model, context[rows], n_actions)
            target = uniform_action_distribution(count, n_actions)
            raw_weights = importance_weights(action[rows], propensity[rows], target)
            for clip in [None, *sorted(set(args.clip_values))]:
                used_weights = importance_weights(action[rows], propensity[rows], target, clip)
                estimates = estimate_policy_value(
                    reward[rows], action[rows], propensity[rows], target, prediction, clip
                )
                intervals = bootstrap_policy_values(
                    reward[rows], action[rows], propensity[rows], target, prediction,
                    clip, args.bootstrap_samples,
                    seed + count + (0 if clip is None else int(clip * 100)),
                )
                for estimator in ("dm", "ips", "snips", "dr"):
                    estimate = estimates[estimator]
                    estimates[f"{estimator}_absolute_error"] = abs(estimate - ground_truth["estimate"])
                    estimates[f"{estimator}_relative_error"] = (
                        abs(estimate - ground_truth["estimate"])
                        / max(abs(ground_truth["estimate"]), 1e-12)
                    )
                experiments.append({
                    "seed": int(seed),
                    "sample_fraction": float(fraction),
                    "evaluation_rounds": int(count),
                    "clip": None if clip is None else float(clip),
                    "estimates": estimates,
                    "bootstrap_intervals": intervals,
                    "importance_weights": importance_weight_diagnostics(raw_weights, used_weights),
                })

    result = {
        "dataset": "Open Bandit Dataset",
        "campaign": args.campaign,
        "behavior_policy": args.behavior_policy,
        "target_policy": "uniform_random",
        "evaluation_contract": {
            "behavior_feedback": "bts propensity-bearing logs",
            "on_policy_ground_truth": "separate random-policy logs",
            "reward_model_fit": "behavior-log half-sample disjoint from OPE evaluation pool",
            "kuairand_propensities": "not assumed or estimated",
        },
        "behavior_rounds": int(len(reward)),
        "on_policy_random_rounds": int(len(random_reward)),
        "on_policy_ground_truth": ground_truth,
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
