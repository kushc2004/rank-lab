import pandas as pd
import pytest

from ranklab.data.splitting import assign_splits, validate_splits
from ranklab.retrieval.popularity import fit_popularity
from ranklab.retrieval.bpr import fit_bpr


def _rows():
    return pd.DataFrame([
        {"user_id": 1, "item_id": 10, "timestamp_ms": 1, "date": 20220421, "is_random": 0, "long_view": 0},
        {"user_id": 1, "item_id": 11, "timestamp_ms": 2, "date": 20220421, "is_random": 0, "long_view": 1},
        {"user_id": 1, "item_id": 12, "timestamp_ms": 3, "date": 20220422, "is_random": 0, "long_view": 1},
        {"user_id": 1, "item_id": 13, "timestamp_ms": 4, "date": 20220501, "is_random": 0, "long_view": 1},
        {"user_id": 2, "item_id": 14, "timestamp_ms": 5, "date": 20220422, "is_random": 1, "long_view": 1},
    ])


def _config():
    return {"train_end": 20220421, "validation_start": 20220422, "validation_end": 20220430, "standard_test_start": 20220501, "standard_test_end": 20220508, "randomized_test_start": 20220422, "randomized_test_end": 20220508}


def test_randomized_rows_are_never_training_data():
    split = assign_splits(_rows(), _config())
    validate_splits(split)
    train = split.loc[split["split"].eq("train")]
    assert len(train) == 2
    assert not train["is_random"].any()
    assert set(split.loc[split["is_random"].eq(1), "split"]) == {"randomized_test"}


def test_official_date_partition_is_valid_when_raw_clocks_cross_a_day_boundary():
    frame = _rows()
    frame.loc[frame["date"].eq(20220422), "timestamp_ms"] = 0
    split = assign_splits(frame, _config())
    validate_splits(split)


def test_models_reject_randomized_training_rows():
    mixed = _rows().iloc[:3].copy()
    mixed.loc[mixed.index[-1], "is_random"] = 1
    with pytest.raises(ValueError, match="standard logs"):
        fit_popularity(mixed)
    with pytest.raises(ValueError, match="standard logs"):
        fit_bpr(mixed, epochs=1)
