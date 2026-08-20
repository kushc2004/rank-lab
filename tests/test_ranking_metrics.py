import pandas as pd

from ranklab.evaluation.ranking_metrics import evaluate_ranked_impressions


def test_ranking_metrics_use_observed_user_date_context_only():
    frame = pd.DataFrame([
        {"user_id": 1, "item_id": 1, "date": 20220501, "long_view": 1, "score": .9},
        {"user_id": 1, "item_id": 2, "date": 20220501, "long_view": 0, "score": .8},
        {"user_id": 2, "item_id": 3, "date": 20220501, "long_view": 0, "score": .7},
    ])
    metrics, groups = evaluate_ranked_impressions(frame, k_values=(1, 2))
    assert metrics["eligible_groups"] == 1
    assert metrics["ndcg@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert set(groups["evaluation_group"]) == {"1:20220501"}
