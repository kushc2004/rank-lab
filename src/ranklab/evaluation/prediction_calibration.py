from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np


@dataclass
class PointwiseProbabilityModel:
    classifier: object
    calibrator: object
    feature_names: tuple[str, ...]

    def predict(self, frame) -> np.ndarray:
        raw = self.classifier.predict_proba(frame[list(self.feature_names)])[:, 1]
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> "PointwiseProbabilityModel":
        with Path(path).open("rb") as source:
            model = pickle.load(source)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a PointwiseProbabilityModel")
        return model


def fit_pointwise_probability(train, validation, feature_names, seed: int = 42) -> PointwiseProbabilityModel:
    """Fit a binary engagement model, then calibrate it only on validation."""
    try:
        from lightgbm import LGBMClassifier
        from sklearn.isotonic import IsotonicRegression
    except ModuleNotFoundError as error:
        raise RuntimeError("Prediction calibration requires pip install -e '.[full]'") from error
    features = tuple(feature_names)
    classifier = LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=30, subsample=0.9,
        colsample_bytree=0.9, random_state=seed, n_jobs=-1, verbosity=-1,
    )
    classifier.fit(train[list(features)], train["label"].gt(0).astype(int))
    raw_validation = classifier.predict_proba(validation[list(features)])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    calibrator.fit(raw_validation, validation["label"].gt(0).astype(int))
    return PointwiseProbabilityModel(classifier, calibrator, features)


def probability_metrics(labels, probabilities, bins: int = 10) -> dict[str, float | int | list]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    brier = float(np.mean(np.square(probabilities - labels)))
    log_loss = float(-np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)))
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        if not selected.any():
            continue
        confidence = float(probabilities[selected].mean())
        accuracy = float(labels[selected].mean())
        share = float(selected.mean())
        ece += share * abs(confidence - accuracy)
        reliability.append({
            "bin": index, "count": int(selected.sum()),
            "mean_probability": confidence, "observed_rate": accuracy,
        })
    classification: dict[str, float] = {}
    if len(np.unique(labels)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score
        classification = {
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "pr_auc": float(average_precision_score(labels, probabilities)),
        }
    return {
        "brier_score": brier,
        "log_loss": log_loss,
        "expected_calibration_error": float(ece),
        "rows": int(len(labels)),
        "reliability": reliability,
        **classification,
    }
