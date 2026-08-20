from __future__ import annotations

import numpy as np
import pandas as pd

def evaluate_ranked_impressions(frame: pd.DataFrame, k_values: tuple[int, ...] = (5, 10, 20), min_group_size: int = 2) -> tuple[dict, pd.DataFrame]:
    # A group is an explicit evaluation context, not a claimed slate/request ID.
    work = frame.copy()
    if "context_id" in work:
        work["evaluation_group"] = work["context_id"].astype(str)
    else:
        work["evaluation_group"] = work.user_id.astype(str) + ":" + work.date.astype(str)
    work["_group_size"] = work.groupby("evaluation_group")["item_id"].transform("size")
    work["_total_positive"] = work.groupby("evaluation_group")["long_view"].transform("sum")
    eligible = work.loc[work["_group_size"].ge(min_group_size) & work["_total_positive"].gt(0)].copy()
    ranked = eligible.sort_values(["evaluation_group", "score", "item_id"], ascending=[True, False, True], kind="stable")
    ranked["_rank"] = ranked.groupby("evaluation_group", sort=False).cumcount() + 1
    ranked["_discounted_label"] = ranked["long_view"] / np.log2(ranked["_rank"] + 1)
    ranked["_positive_rank"] = ranked["_rank"].where(ranked["long_view"].eq(1))
    group_meta = ranked.groupby("evaluation_group", sort=False).agg(user_id=("user_id", "first"), date=("date", "first"), total_positive=("_total_positive", "first"))
    records = []
    for k in k_values:
        at_k = ranked.loc[ranked["_rank"].le(k)]
        aggregate = at_k.groupby("evaluation_group", sort=False).agg(dcg=("_discounted_label", "sum"), hits=("long_view", "sum"), first_positive_rank=("_positive_rank", "min"))
        result = group_meta.join(aggregate, how="left").fillna({"dcg": 0.0, "hits": 0.0})
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        result["idcg"] = [discounts[:min(k, int(positive))].sum() for positive in result["total_positive"]]
        result["ndcg"] = result["dcg"] / result["idcg"]
        result["recall"] = result["hits"] / result["total_positive"]
        result["mrr"] = np.where(result["first_positive_rank"].notna(), 1.0 / result["first_positive_rank"], 0.0)
        result["k"] = k
        records.append(result.reset_index()[["evaluation_group", "user_id", "date", "k", "ndcg", "recall", "mrr"]])
    per_group = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    metrics = {f"{metric}@{k}": float(per_group.loc[per_group.k.eq(k), metric].mean()) for k in k_values for metric in ("ndcg", "recall", "mrr")} if not per_group.empty else {}
    metrics["eligible_groups"] = int(eligible.evaluation_group.nunique()) if not eligible.empty else 0
    metrics["eligible_rows"] = int(len(eligible))
    return metrics, per_group
