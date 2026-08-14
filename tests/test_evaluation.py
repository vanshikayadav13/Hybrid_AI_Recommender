"""
Unit Tests for Recommendation Evaluation Metrics Module
"""

from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k


def test_precision_at_k():
    recs = ["P1", "P2", "P3", "P4", "P5"]
    relevant = {"P2", "P4", "P8"}
    
    # Precision@3: recs[:3] -> ["P1", "P2", "P3"]. Intersect {"P2"} = 1 hit. Precision = 1 / 3 = 0.3333
    p3 = precision_at_k(recs, relevant, k=3)
    assert round(p3, 4) == 0.3333

    # Precision@5: recs[:5] -> ["P1", "P2", "P3", "P4", "P5"]. Intersect {"P2", "P4"} = 2 hits. Precision = 2 / 5 = 0.4
    p5 = precision_at_k(recs, relevant, k=5)
    assert p5 == 0.4


def test_recall_at_k():
    recs = ["P1", "P2", "P3", "P4", "P5"]
    relevant = {"P2", "P4", "P8"}
    
    # Recall@5: 2 hits out of 3 relevant items -> 2 / 3 = 0.6667
    r5 = recall_at_k(recs, relevant, k=5)
    assert round(r5, 4) == 0.6667


def test_ndcg_at_k():
    # Ideal ranking: relevant item at rank 1
    recs_perfect = ["P2", "P4", "P1"]
    relevant = {"P2", "P4"}
    
    # Perfect ranking NDCG@2 should be 1.0
    ndcg_perfect = ndcg_at_k(recs_perfect, relevant, k=2)
    assert ndcg_perfect == 1.0

    # Non-ideal ranking: relevant item at rank 2
    recs_suboptimal = ["P1", "P2", "P3"]
    # DCG@2 = 0 + 1 / log2(3) = 0.6309. IDCG@2 = 1/log2(2) + 1/log2(3) = 1.6309. NDCG@2 = 0.6309 / 1.6309 = 0.38685
    ndcg_sub = ndcg_at_k(recs_suboptimal, relevant, k=2)
    assert 0.35 < ndcg_sub < 0.40
