"""
Unit & Integration Tests for Multi-Seed Evaluation Engine
"""

import pytest
import pandas as pd
from src.evaluation.multi_seed_evaluator import MultiSeedEvaluator, DEFAULT_EVAL_SEEDS


@pytest.fixture
def sample_catalog() -> pd.DataFrame:
    data = [
        {"product_id": "P101", "product_name": "Wireless Headphones", "category": "Electronics", "description": "Noise canceling audio headset.", "price": 150.0, "rating": 4.5},
        {"product_id": "P102", "product_name": "Bluetooth Earbuds", "category": "Electronics", "description": "In ear wireless sound.", "price": 50.0, "rating": 4.0},
        {"product_id": "P103", "product_name": "Running Shoes", "category": "Apparel", "description": "Athletic sneakers for jogging.", "price": 80.0, "rating": 4.2},
        {"product_id": "P104", "product_name": "Smart Fitness Watch", "category": "Electronics", "description": "Digital watch with heart monitor.", "price": 120.0, "rating": 4.6},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    data = [
        {"user_id": "U1", "product_id": "P101", "user_rating": 5.0},
        {"user_id": "U1", "product_id": "P102", "user_rating": 4.0},
        {"user_id": "U2", "product_id": "P102", "user_rating": 4.5},
        {"user_id": "U2", "product_id": "P103", "user_rating": 5.0},
        {"user_id": "U3", "product_id": "P101", "user_rating": 4.0},
        {"user_id": "U3", "product_id": "P104", "user_rating": 5.0},
    ]
    return pd.DataFrame(data)


def test_empty_seeds_raises_value_error(sample_catalog: pd.DataFrame, sample_interactions: pd.DataFrame):
    with pytest.raises(ValueError) as exc:
        MultiSeedEvaluator(sample_catalog, sample_interactions, seeds=[])
    assert "Seeds list cannot be empty" in str(exc.value)


def test_calculate_summary_stats():
    values = [0.10, 0.20, 0.30, 0.40, 0.50]
    stats = MultiSeedEvaluator.calculate_summary_stats(values)

    assert stats["mean"] == 0.30
    assert stats["min"] == 0.10
    assert stats["max"] == 0.50
    assert "std" in stats
    assert "ci_95" in stats


def test_multi_seed_evaluation_run(sample_catalog: pd.DataFrame, sample_interactions: pd.DataFrame):
    evaluator = MultiSeedEvaluator(sample_catalog, sample_interactions, seeds=[42, 7])
    configs = {
        "Test Config 1": {"content": 0.5, "collaborative": 0.5, "nlp": 0.0, "image": 0.0}
    }

    df = evaluator.run_multi_seed_evaluation(configs, top_k=2)
    assert not df.empty
    assert "P@K_Mean" in df.columns
    assert "R@K_Mean" in df.columns
    assert "NDCG@K_Mean" in df.columns
    assert "P@K_95CI" in df.columns


def test_ablation_study_run(sample_catalog: pd.DataFrame, sample_interactions: pd.DataFrame):
    evaluator = MultiSeedEvaluator(sample_catalog, sample_interactions, seeds=[42])
    ablation_df = evaluator.run_ablation_study(top_k=2)

    assert not ablation_df.empty
    assert len(ablation_df) == 5
    assert "1. Content Only" in ablation_df["Configuration"].values
    assert "5. Full Hybrid (All Signals)" in ablation_df["Configuration"].values


def test_signal_toggle_analysis(sample_catalog: pd.DataFrame, sample_interactions: pd.DataFrame):
    evaluator = MultiSeedEvaluator(sample_catalog, sample_interactions, seeds=[42])
    toggles = evaluator.run_signal_toggle_analysis(top_k=2)

    assert "image_comparison" in toggles
    assert "nlp_comparison" in toggles
    assert not toggles["image_comparison"].empty
    assert not toggles["nlp_comparison"].empty
