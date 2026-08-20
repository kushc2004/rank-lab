from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DOMAIN_FEATURES = (
    "tab", "hour", "day_of_week", "session_position",
    "hist_item_exposure", "hist_item_reward_rate", "hist_user_exposure",
    "hist_user_long_view_rate",
)


@dataclass
class DensityRatioModel:
    pipeline: object
    features: tuple[str, ...]
    train_random_prior: float

    def probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.features) - set(frame.columns))
        if missing:
            raise ValueError(f"density-ratio input is missing features: {missing}")
        probabilities = self.pipeline.predict_proba(frame[list(self.features)])[:, 1]
        if not np.isfinite(probabilities).all():
            raise ValueError("density-ratio classifier produced non-finite probabilities")
        return probabilities

    def weights(
        self,
        frame: pd.DataFrame,
        clip_max: float | None = 10.0,
        self_normalize: bool = False,
    ) -> np.ndarray:
        q = np.clip(self.probabilities(frame), 1e-5, 1 - 1e-5)
        # Classifier odds include its training-domain prior. Correcting by
        # P(D=0)/P(D=1) recovers p_random(x)/p_standard(x).
        prior_correction = (1 - self.train_random_prior) / self.train_random_prior
        weight = (q / (1 - q)) * prior_correction
        if clip_max is not None:
            if not np.isfinite(clip_max) or clip_max <= 0:
                raise ValueError("clip_max must be positive and finite when provided")
            weight = np.minimum(weight, clip_max)
        if not np.isfinite(weight).all() or np.any(weight < 0):
            raise ValueError("density-ratio weights must be finite and non-negative")
        if self_normalize:
            mean = float(weight.mean())
            if mean <= 0:
                raise ValueError("cannot self-normalize density-ratio weights with zero mean")
            weight = weight / mean
        return weight

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> "DensityRatioModel":
        with Path(path).open("rb") as source:
            model = pickle.load(source)
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a DensityRatioModel")
        return model


def fit_density_ratio(
    standard: pd.DataFrame,
    randomized: pd.DataFrame,
    features: tuple[str, ...] = DEFAULT_DOMAIN_FEATURES,
    seed: int = 42,
    max_rows_per_domain: int = 300_000,
) -> tuple[DensityRatioModel, dict[str, object]]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ModuleNotFoundError as error:
        raise RuntimeError("Density-ratio training requires pip install -e '.[full]'") from error
    present = tuple(column for column in features if column in standard and column in randomized)
    if not present:
        raise ValueError("no common exposure-time domain features")
    if standard.empty or randomized.empty:
        raise ValueError("both standard and randomized adaptation domains must contain rows")
    if max_rows_per_domain < 2:
        raise ValueError("max_rows_per_domain must be at least 2")
    rng = np.random.default_rng(seed)
    sample = lambda frame: frame.sample(
        min(len(frame), max_rows_per_domain), random_state=int(rng.integers(2**31 - 1))
    )
    left, right = sample(standard).copy(), sample(randomized).copy()
    left["_domain"] = 0
    right["_domain"] = 1
    data = pd.concat([left, right], ignore_index=True)
    if data["_domain"].value_counts().min() < 2:
        raise ValueError("each exposure domain needs at least two rows")
    train, validation = train_test_split(
        data,
        test_size=0.2,
        random_state=seed,
        stratify=data["_domain"],
    )
    categorical = [column for column in present if column == "tab"]
    numeric = [column for column in present if column not in categorical]
    transformer = ColumnTransformer(
        [
            ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=5)), categorical),
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ]
    )
    pipeline = make_pipeline(transformer, LogisticRegression(max_iter=300, solver="saga", n_jobs=-1, random_state=seed))
    pipeline.fit(train[list(present)], train["_domain"])
    q = pipeline.predict_proba(validation[list(present)])[:, 1]
    model = DensityRatioModel(pipeline, present, float(train["_domain"].mean()))
    weights = model.weights(left, clip_max=None)
    diagnostics = weight_diagnostics(weights)
    diagnostics.update(
        {
            "domain_auc": float(roc_auc_score(validation["_domain"], q)),
            "domain_train_random_prior": model.train_random_prior,
            "standard_rows_sampled": int(len(left)),
            "randomized_rows_sampled": int(len(right)),
            "features": list(present),
            "feature_contract": (
                "exposure-time context and strict pre-context aggregates only; "
                "raw user/item identifiers and undated snapshot features are excluded"
            ),
        }
    )
    return model, diagnostics


def weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weight diagnostics require a non-empty one-dimensional array")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weight diagnostics require finite non-negative weights")
    total = weights.sum()
    squared_total = np.square(weights).sum()
    return {
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_max": float(weights.max()),
        "weight_p01": float(np.percentile(weights, 1)),
        "weight_p50": float(np.percentile(weights, 50)),
        "weight_p99": float(np.percentile(weights, 99)),
        "effective_sample_size": float(total * total / squared_total) if squared_total > 0 else 0.0,
    }
