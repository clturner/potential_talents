import pandas as pd
import numpy as np
from itertools import chain

def evaluate_reranking(df, starred_ids, rank_col="fit_BERT_reRanked", k=None):
    """
    Evaluate reranking quality using starred candidates and return ranked unstarred list.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with 'id' and ranking column.
    - starred_ids (list): List of starred candidate IDs (previously hired).
    - rank_col (str): Column to sort by for ranking (default: "fit_BERT_reRanked").
    - k (int or None): Top k positions to evaluate; if None, uses number of starred candidates (default: None).
    
    Returns:
    - dict: Evaluation metrics (Precision@k, Mean Rank, MRR, NDCG@k).
    - pd.DataFrame: Ranked DataFrame of unstarred candidates.
    """
    # Flatten starred_ids if nested
    starred_ids = list(chain.from_iterable([x] if not isinstance(x, (list, tuple)) else x for x in starred_ids))

    # Convert starred_ids to match df["id"] dtype
    id_dtype = df["id"].dtype
    if np.issubdtype(id_dtype, np.integer):
        starred_ids = [int(x) for x in starred_ids]
    elif np.issubdtype(id_dtype, np.object_) or str(id_dtype) == "string":
        starred_ids = [str(x) for x in starred_ids]

    # Sort by rank_col in descending order
    df_sorted = df.sort_values(by=rank_col, ascending=False).reset_index(drop=True)
    df_sorted["rank"] = np.arange(1, len(df_sorted) + 1)

    # Identify starred candidates
    starred_mask = df_sorted["id"].isin(starred_ids)
    num_starred = len(set(starred_ids) & set(df_sorted["id"]))  # Actual matches in df

    # Metrics
    top_k_starred = df_sorted["id"].head(k).isin(starred_ids).sum()
    precision_at_k = top_k_starred / k if k > 0 else 0

    starred_ranks = df_sorted[starred_mask]["rank"]
    mean_rank_starred = starred_ranks.mean() if not starred_ranks.empty else float("inf")

    first_starred_rank = starred_ranks.min() if not starred_ranks.empty else float("inf")
    mrr = 1 / first_starred_rank if first_starred_rank != float("inf") else 0

    relevance = np.where(starred_mask, 1, 0)
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance[:k]))
    ideal_relevance = [1] * min(num_starred, k) + [0] * max(0, k - num_starred)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0

    metrics = {
        f"precision_at_{k}": precision_at_k,
        "mean_rank_starred": mean_rank_starred,
        "mrr": mrr,
        f"ndcg_at_{k}": ndcg_at_k
    }

    # Filter out starred candidates
    df_unstarred = df_sorted[~starred_mask].drop(columns=["rank"]).reset_index(drop=True)

    return metrics, df_unstarred

    



