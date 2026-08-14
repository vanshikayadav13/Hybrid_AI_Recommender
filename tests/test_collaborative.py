"""
Unit Tests for Collaborative Filtering & Cold Start Engine
"""

import pytest
import pandas as pd
from src.models.collaborative_filtering import CollaborativeRecommender
from src.models.cold_start import ColdStartHandler, RecommendationEngine
from src.models.content_based import ContentBasedRecommender


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    data = [
        {"user_id": "U1", "product_id": "P101", "user_rating": 5.0},
        {"user_id": "U1", "product_id": "P102", "user_rating": 4.0},
        {"user_id": "U2", "product_id": "P101", "user_rating": 4.5},
        {"user_id": "U2", "product_id": "P103", "user_rating": 5.0},
        {"user_id": "U3", "product_id": "P102", "user_rating": 4.0},
        {"user_id": "U3", "product_id": "P103", "user_rating": 4.0},
        {"user_id": "U3", "product_id": "P104", "user_rating": 5.0},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_products() -> pd.DataFrame:
    data = [
        {"product_id": "P101", "product_name": "Headphones", "category": "Electronics", "description": "Over ear noise canceling", "price": 100.0, "rating": 4.5},
        {"product_id": "P102", "product_name": "Earbuds", "category": "Electronics", "description": "Wireless bluetooth earbuds", "price": 50.0, "rating": 4.0},
        {"product_id": "P103", "product_name": "T-Shirt", "category": "Apparel", "description": "Cotton graphic tee", "price": 20.0, "rating": 4.2},
        {"product_id": "P104", "product_name": "Watch", "category": "Electronics", "description": "Smart fitness watch", "price": 150.0, "rating": 4.8},
    ]
    return pd.DataFrame(data)


def test_collaborative_recommendations(sample_interactions: pd.DataFrame, sample_products: pd.DataFrame):
    cf = CollaborativeRecommender().fit(sample_interactions, sample_products)
    
    # 1. Recommendations for U1 (Interacted with P101, P102) -> should recommend P103 or P104
    recs_u1 = cf.recommend_collaborative_products("U1", top_k=2)
    assert not recs_u1.empty
    rec_pids = recs_u1["product_id"].tolist()
    
    # Exclude already interacted products
    assert "P101" not in rec_pids
    assert "P102" not in rec_pids
    
    # Score ordering check
    scores = recs_u1["predicted_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_unknown_user_cold_start(sample_interactions: pd.DataFrame, sample_products: pd.DataFrame):
    cf = CollaborativeRecommender().fit(sample_interactions, sample_products)
    cb = ContentBasedRecommender().fit(sample_products)
    engine = RecommendationEngine(content_model=cb, collaborative_model=cf, products_df=sample_products)
    
    # Unknown user should trigger popularity cold-start strategy
    cold_recs = engine.recommend_for_user("UNKNOWN_USER_777", top_k=2)
    assert len(cold_recs) == 2
    assert "Cold-Start Popularity Strategy" in cold_recs.iloc[0]["recommendation_method"]
    # Highest rated product is P104 (4.8)
    assert cold_recs.iloc[0]["product_id"] == "P104"
