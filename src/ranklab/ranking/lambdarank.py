from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ranklab.ranking.ranker_features import FEATURE_COLUMNS


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["context_id", "item_id"], kind="stable").reset_index(drop=True)


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    ordered = _ordered(frame)
    sizes = ordered.groupby("context_id", sort=False).size().to_numpy(dtype=np.int32)
    if sizes.sum() != len(ordered):
        raise AssertionError("ranking group sizes do not sum to row count")
    return sizes


@dataclass
class LambdaRankModel:
    booster: object
    feature_names: tuple[str, ...] = FEATURE_COLUMNS

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(frame[list(self.feature_names)]), dtype=float)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.booster_.save_model(str(path))

    def feature_importance(self) -> pd.DataFrame:
        booster = self.booster.booster_ if hasattr(self.booster, "booster_") else self.booster
        return pd.DataFrame(
            {
                "feature": list(self.feature_names),
                "gain": booster.feature_importance(importance_type="gain"),
                "split": booster.feature_importance(importance_type="split"),
            }
        ).sort_values(["gain", "split"], ascending=False, kind="stable")

    @classmethod
    def load(cls, path: str | Path) -> "LambdaRankModel":
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as error:
            raise RuntimeError("Loading LambdaRank requires pip install -e '.[full]'") from error
        return cls(lgb.Booster(model_file=str(path)))


def fit_lambdarank(
    train: pd.DataFrame,
    validation: pd.DataFrame | None = None,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
    n_estimators: int = 300,
    early_stopping_rounds: int = 30,
) -> LambdaRankModel:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as error:
        raise RuntimeError("LambdaRank requires the full extra: pip install -e '.[full]'") from error
    working = train.copy()
    if sample_weight is not None:
        if len(sample_weight) != len(working):
            raise ValueError("sample_weight must have one value per training row")
        working["_sample_weight"] = np.asarray(sample_weight, dtype=float)
    ordered = _ordered(working)
    if sample_weight is not None:
        weights = ordered.pop("_sample_weight").to_numpy()
    else:
        weights = None
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", eval_at=[5, 10, 20],
        lambdarank_truncation_level=13, n_estimators=n_estimators,
        learning_rate=0.05, num_leaves=31, min_child_samples=30,
        subsample=0.9, colsample_bytree=0.9, random_state=seed,
        n_jobs=-1, verbosity=-1,
    )
    kwargs = {}
    if validation is not None and not validation.empty:
        val = _ordered(validation)
        callbacks = []
        if early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(early_stopping_rounds, verbose=False)
            )
        kwargs.update(
            eval_set=[(val[list(FEATURE_COLUMNS)], val["label"].astype(int))],
            eval_group=[group_sizes(val)],
            callbacks=callbacks,
        )
    model.fit(
        ordered[list(FEATURE_COLUMNS)], ordered["label"].astype(int),
        group=group_sizes(ordered), sample_weight=weights, **kwargs,
    )
    return LambdaRankModel(model)
