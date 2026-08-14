"""
Unit Tests for Content-Based Recommendation Engine
"""

import pytest
import pandas as pd
from src.models.content_based import ContentBasedRecommender


@pytest.fixture
def sample_catalog() -> pd.DataFrame:
    data = [
        {
            "product_id": "P101",
            "product_name": "Wireless Headphones Noise Canceling",
            "category": "Electronics",
            "description": "Over-ear Bluetooth headphones with active noise cancellation.",
            "price": 199.99,
            "rating": 4.5
        },
        {
            "product_id": "P102",
            "product_name": "Wireless Bluetooth Earbuds",
            "category": "Electronics",
            "description": "In-ear wireless earbuds with charging case.",
            "price": 49.99,
            "rating": 4.2
        },
        {
            "product_id": "P103",
            "product_name": "Cotton Graphic T-Shirt",
            "category": "Apparel",
            "description": "Casual unisex graphic tee made of 100% organic cotton.",
            "price": 24.99,
            "rating": 4.0
        },
        {
            "product_id": "P104",
            "product_name": "Smart Fitness Watch",
            "category": "Electronics",
            "description": "Digital watch with heart rate monitor and wireless Bluetooth.",
            "price": 129.99,
            "rating": 4.6
        }
    ]
    return pd.DataFrame(data)


def test_content_based_recommendation(sample_catalog: pd.DataFrame):
    cb = ContentBasedRecommender().fit(sample_catalog)
    
    # 1. Valid recommendation for P101
    recs = cb.recommend_similar_products("P101", top_k=2)
    assert len(recs) == 2
    assert "product_id" in recs.columns
    assert "similarity_score" in recs.columns
    
    # 2. Input product exclusion check (P101 must NOT be in its own recommendations)
    assert "P101" not in recs["product_id"].tolist()
    
    # 3. Check similarity ordering (descending)
    scores = recs["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    
    # 4. Electronic items (P102, P104) should rank higher than T-Shirt (P103)
    top_recommended_id = recs.iloc[0]["product_id"]
    assert top_recommended_id in ["P102", "P104"]


def test_invalid_product_id_handling(sample_catalog: pd.DataFrame):
    cb = ContentBasedRecommender().fit(sample_catalog)
    
    # Invalid product ID should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        cb.recommend_similar_products("INVALID_ID_999", top_k=3)
    assert "not found in catalog" in str(exc_info.value)


def test_top_k_parameter(sample_catalog: pd.DataFrame):
    cb = ContentBasedRecommender().fit(sample_catalog)
    recs_k1 = cb.recommend_similar_products("P101", top_k=1)
    recs_k3 = cb.recommend_similar_products("P101", top_k=3)
    
    assert len(recs_k1) == 1
    assert len(recs_k3) == 3
