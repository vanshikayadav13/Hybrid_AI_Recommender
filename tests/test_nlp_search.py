"""
Unit Tests for NLP Semantic Search Module
"""

import pytest
import pandas as pd
from pathlib import Path
from src.models.nlp_search import NLPSemanticSearch


@pytest.fixture
def sample_products() -> pd.DataFrame:
    data = [
        {
            "product_id": "P101",
            "product_name": "Noise-Canceling Wireless Headphones",
            "category": "Electronics",
            "description": "Over-ear Bluetooth headphones with active noise cancellation for music and gaming.",
            "price": 199.99,
            "rating": 4.5
        },
        {
            "product_id": "P102",
            "product_name": "Breathable Athletic Running Shoes",
            "category": "Apparel",
            "description": "Lightweight mesh sneakers designed for marathon running and daily workouts.",
            "price": 89.99,
            "rating": 4.3
        },
        {
            "product_id": "P103",
            "product_name": "Automatic Espresso Machine",
            "category": "Home & Kitchen",
            "description": "Compact coffee maker with integrated milk frother.",
            "price": 299.99,
            "rating": 4.7
        }
    ]
    return pd.DataFrame(data)


def test_nlp_embedding_dimension(sample_products: pd.DataFrame):
    nlp = NLPSemanticSearch().fit(sample_products)
    assert nlp.product_embeddings is not None
    # all-MiniLM-L6-v2 produces 384-dimensional dense vectors
    assert nlp.product_embeddings.shape == (3, 384)


def test_nlp_semantic_search_queries(sample_products: pd.DataFrame):
    nlp = NLPSemanticSearch().fit(sample_products)
    
    # 1. Normal semantic query
    query = "headset for gaming and audio"
    results = nlp.semantic_search(query, top_k=2)
    assert len(results) == 2
    assert "product_id" in results.columns
    assert "similarity_score" in results.columns
    
    # Top result for audio/gaming headset should be P101 (Headphones)
    assert results.iloc[0]["product_id"] == "P101"
    
    # Check similarity ordering (descending)
    scores = results["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_empty_and_unknown_query(sample_products: pd.DataFrame):
    nlp = NLPSemanticSearch().fit(sample_products)
    
    # Empty query should return empty DataFrame gracefully
    empty_res = nlp.semantic_search("", top_k=5)
    assert empty_res.empty
    
    # Out of vocabulary / unknown words query should process without error
    unknown_res = nlp.semantic_search("xyzqwerty unkwnword123", top_k=2)
    assert not unknown_res.empty
    assert len(unknown_res) == 2


def test_nlp_model_persistence(sample_products: pd.DataFrame, tmp_path: Path):
    nlp = NLPSemanticSearch().fit(sample_products)
    save_file = tmp_path / "nlp_search.joblib"
    nlp.save_model(save_file)
    
    # Load model back
    loaded_nlp = NLPSemanticSearch.load_model(save_file)
    assert loaded_nlp.product_embeddings.shape == (3, 384)
    res = loaded_nlp.semantic_search("running shoes", top_k=1)
    assert res.iloc[0]["product_id"] == "P102"
