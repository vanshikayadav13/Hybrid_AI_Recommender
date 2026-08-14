"""
Recommendation Evaluation Metrics Module

Provides mathematical evaluation functions for top-K recommendation lists:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Evaluation pipeline framework over train/test split interaction sets.
"""

from typing import Dict, List, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def precision_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """
    Computes Precision@K: Fraction of top-K recommendations that are relevant.

    Formula:
        Precision@K = |Recommended@K intersect Relevant| / K
    """
    if k <= 0:
        return 0.0
    rec_k = recommended_items[:k]
    hits = len(set(rec_k).intersection(relevant_items))
    return float(hits / k)


def recall_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """
    Computes Recall@K: Fraction of ground-truth relevant items captured in top-K recommendations.

    Formula:
        Recall@K = |Recommended@K intersect Relevant| / |Relevant|
    """
    if not relevant_items or k <= 0:
        return 0.0
    rec_k = recommended_items[:k]
    hits = len(set(rec_k).intersection(relevant_items))
    return float(hits / len(relevant_items))


def ndcg_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
    """
    Computes NDCG@K: Normalized Discounted Cumulative Gain at rank K.

    Formula:
        DCG@K = sum_{i=1}^K rel_i / log2(i + 1)
        IDCG@K = sum_{i=1}^min(K, |Relevant|) 1 / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K
    """
    if not relevant_items or k <= 0:
        return 0.0

    rec_k = recommended_items[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k, start=1):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 1)

    # Ideal DCG assumes all top relevant items appear at the very top positions
    idcg = 0.0
    n_ideal = min(k, len(relevant_items))
    for i in range(1, n_ideal + 1):
        idcg += 1.0 / np.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def train_test_split_interactions(
    interactions_df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits user interaction logs into Train and Test dataframes (per-user split).

    Args:
        interactions_df (pd.DataFrame): Input interactions dataframe [user_id, product_id, user_rating].
        test_ratio (float): Fraction of interactions per user to hold out for testing.
        seed (int): Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    np.random.seed(seed)
    train_list = []
    test_list = []

    for user_id, group in interactions_df.groupby("user_id"):
        if len(group) < 2:
            train_list.append(group)
            continue
        
        shuffled = group.sample(frac=1.0, random_state=seed)
        n_test = max(1, int(len(shuffled) * test_ratio))
        test_list.append(shuffled.iloc[:n_test])
        train_list.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame(columns=interactions_df.columns)
    
    return train_df, test_df


def evaluate_recommender(
    recommender,
    test_df: pd.DataFrame,
    k_list: List[int] = [5, 10],
    threshold_rating: float = 3.0
) -> pd.DataFrame:
    """
    Evaluates a Collaborative or Hybrid Recommender on test interaction sets.

    Args:
        recommender: Fitted recommender object with recommend_collaborative_products or recommend_for_user method.
        test_df (pd.DataFrame): Held-out test interaction dataframe [user_id, product_id, user_rating].
        k_list (List[int]): List of K cutoff values to evaluate (e.g. [5, 10]).
        threshold_rating (float): Minimum rating to consider an item "relevant".

    Returns:
        pd.DataFrame: Table of averaged Precision@K, Recall@K, and NDCG@K scores.
    """
    # Filter ground truth relevant items in test set
    relevant_test = test_df[test_df["user_rating"] >= threshold_rating]
    user_ground_truth = relevant_test.groupby("user_id")["product_id"].apply(set).to_dict()

    eval_results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_list}

    for user_id, rel_items in user_ground_truth.items():
        try:
            if hasattr(recommender, "recommend_collaborative_products"):
                recs_df = recommender.recommend_collaborative_products(user_id, top_k=max(k_list))
            elif hasattr(recommender, "recommend_for_user"):
                recs_df = recommender.recommend_for_user(user_id, top_k=max(k_list))
            else:
                continue

            if recs_df.empty or "product_id" not in recs_df.columns:
                rec_items = []
            else:
                rec_items = recs_df["product_id"].tolist()

            for k in k_list:
                eval_results[k]["precision"].append(precision_at_k(rec_items, rel_items, k))
                eval_results[k]["recall"].append(recall_at_k(rec_items, rel_items, k))
                eval_results[k]["ndcg"].append(ndcg_at_k(rec_items, rel_items, k))

        except Exception as err:
            logger.warning("Error evaluating user %s: %s", user_id, err)

    results_table = []
    for k in k_list:
        p_mean = np.mean(eval_results[k]["precision"]) if eval_results[k]["precision"] else 0.0
        r_mean = np.mean(eval_results[k]["recall"]) if eval_results[k]["recall"] else 0.0
        n_mean = np.mean(eval_results[k]["ndcg"]) if eval_results[k]["ndcg"] else 0.0

        results_table.append({
            "Metric Cutoff (K)": f"K = {k}",
            "Precision@K": round(float(p_mean), 4),
            "Recall@K": round(float(r_mean), 4),
            "NDCG@K": round(float(n_mean), 4)
        })

    return pd.DataFrame(results_table)
