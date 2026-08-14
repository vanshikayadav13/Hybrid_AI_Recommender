"""
Unit Tests for Hybrid Recommender Engine Module
"""

import pytest
import pandas as pd
from pathlib import Path
from src.models import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    NLPSemanticSearch,
    CNNImageSimilarity,
    HybridRecommender,
    DEFAULT_WEIGHTS
)


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
    ]
    return pd.DataFrame(data)


def test_weight_validation():
    # Valid weights (sum = 1.0)
    valid_w = {"content": 0.25, "collaborative": 0.35, "nlp": 0.25, "image": 0.15}
    assert HybridRecommender.validate_weights(valid_w) == valid_w

    # Invalid weights sum != 1.0
    with pytest.raises(ValueError) as exc1:
        HybridRecommender.validate_weights({"content": 0.5, "collaborative": 0.5, "nlp": 0.5, "image": 0.5})
    assert "must sum to 1.0" in str(exc1.value)

    # Negative weight
    with pytest.raises(ValueError) as exc2:
        HybridRecommender.validate_weights({"content": -0.1, "collaborative": 0.6, "nlp": 0.3, "image": 0.2})
    assert "cannot be negative" in str(exc2.value)


def test_min_max_normalization():
    scores = pd.Series([10.0, 20.0, 30.0, 50.0])
    norm = HybridRecommender.min_max_normalize(scores)
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert norm.iloc[0] == 0.0
    assert norm.iloc[-1] == 1.0


def test_hybrid_recommendation_flow(sample_catalog: pd.DataFrame, sample_interactions: pd.DataFrame):
    cb = ContentBasedRecommender().fit(sample_catalog)
    cf = CollaborativeRecommender().fit(sample_interactions, sample_catalog)
    nlp = NLPSemanticSearch().fit(sample_catalog)
    img = CNNImageSimilarity().fit(sample_catalog)

    hybrid = HybridRecommender(
        content_model=cb,
        collaborative_model=cf,
        nlp_model=nlp,
        image_model=img,
        products_df=sample_catalog
    )

    # Test hybrid recommendation for U1 with reference product P101
    recs = hybrid.hybrid_recommend(
        user_id="U1",
        product_id="P101",
        query="wireless audio headphones",
        top_k=2
    )

    assert not recs.empty
    assert len(recs) <= 2
    assert "final_hybrid_score" in recs.columns
    assert "explanation" in recs.columns

    # Check deduplication & interacted item exclusion (U1 interacted with P101, P102; reference P101)
    rec_pids = recs["product_id"].tolist()
    assert "P101" not in rec_pids

    # Ranking score ordering check
    scores = recs["final_hybrid_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_cold_start_and_unknown_handling(sample_catalog: pd.DataFrame, sample_interactions: pd.DataFrame):
    cb = ContentBasedRecommender().fit(sample_catalog)
    cf = CollaborativeRecommender().fit(sample_interactions, sample_catalog)
    nlp = NLPSemanticSearch().fit(sample_catalog)
    img = CNNImageSimilarity().fit(sample_catalog)

    hybrid = HybridRecommender(
        content_model=cb,
        collaborative_model=cf,
        nlp_model=nlp,
        image_model=img,
        products_df=sample_catalog
    )

    # Unknown user should trigger cold-start popularity strategy
    recs_unknown_user = hybrid.hybrid_recommend(user_id="UNKNOWN_USER_999", top_k=2)
    assert not recs_unknown_user.empty

    # Empty query should process safely without crashing
    recs_empty_query = hybrid.hybrid_recommend(query="", top_k=2)
    assert not recs_empty_query.empty


def test_hybrid_model_persistence(sample_catalog: pd.DataFrame, tmp_path: Path):
    cb = ContentBasedRecommender().fit(sample_catalog)
    hybrid = HybridRecommender(content_model=cb, products_df=sample_catalog)
    
    save_file = tmp_path / "hybrid_recommender.joblib"
    hybrid.save_model(save_file)
    
    loaded_hybrid = HybridRecommender.load_model(save_file, content_model=cb, products_df=sample_catalog)
    assert loaded_hybrid.weights == DEFAULT_WEIGHTS
