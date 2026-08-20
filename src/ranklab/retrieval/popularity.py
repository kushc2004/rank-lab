from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class PopularityModel:
    scores: dict[int, float]
    default_score: float

    def predict(self, _user_ids: pd.Series, item_ids: pd.Series) -> np.ndarray:
        return item_ids.map(self.scores).fillna(self.default_score).to_numpy(dtype=float)

def fit_popularity(train: pd.DataFrame, alpha: float = 20.0) -> PopularityModel:
    if train.is_random.any():
        raise ValueError("Popularity baseline may train only on standard logs")
    totals = train.groupby("item_id").long_view.agg(["sum", "count"])
    rate = float(train.long_view.mean())
    scores = ((totals["sum"] + alpha * rate) / (totals["count"] + alpha)).to_dict()
    return PopularityModel({int(k): float(v) for k, v in scores.items()}, rate)
