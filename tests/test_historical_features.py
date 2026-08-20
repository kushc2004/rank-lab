import pandas as pd

from ranklab.features.historical import build_historical_features


def test_historical_features_have_strict_time_cutoff_and_hide_same_timestamp_labels():
    frame = pd.DataFrame([
        {"user_id": 1, "item_id": 10, "timestamp_ms": 100, "long_view": 1},
        {"user_id": 2, "item_id": 10, "timestamp_ms": 100, "long_view": 0},
        {"user_id": 1, "item_id": 10, "timestamp_ms": 101, "long_view": 0},
    ])
    result = build_historical_features(frame, alpha=2.0)
    at_first_timestamp = result.loc[result["timestamp_ms"].eq(100)]
    assert (at_first_timestamp["hist_item_exposure"] == 0).all()
    assert (result["feature_cutoff_ms"] < result["timestamp_ms"]).all()
    later = result.loc[result["timestamp_ms"].eq(101)].iloc[0]
    assert later["hist_item_exposure"] == 2
    assert later["hist_user_exposure"] == 1
