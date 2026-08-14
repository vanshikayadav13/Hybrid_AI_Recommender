"""
Unit & Integration Tests for Flask REST API Module
"""

import pytest
from flask.testing import FlaskClient
from src.api.app import create_app


@pytest.fixture
def client() -> FlaskClient:
    """Fixture initializing Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_endpoint(client: FlaskClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["status"] == "ok"
    assert "total_catalog_items" in data


def test_get_products_list(client: FlaskClient):
    response = client.get("/api/products?page=1&limit=5")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert len(data["data"]) == 5
    assert "pagination" in data
    assert data["pagination"]["page"] == 1


def test_get_products_category_filter(client: FlaskClient):
    response = client.get("/api/products?category=Electronics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    for item in data["data"]:
        assert item["category"].lower() == "electronics"


def test_get_product_by_id_success(client: FlaskClient):
    response = client.get("/api/products/PROD_0101")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["product"]["product_id"] == "PROD_0101"


def test_get_product_by_id_not_found(client: FlaskClient):
    response = client.get("/api/products/PROD_INVALID_9999")
    assert response.status_code == 404
    data = response.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "PRODUCT_NOT_FOUND"


def test_semantic_search_success(client: FlaskClient):
    payload = {"query": "wireless headphones", "top_k": 5}
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["query"] == "wireless headphones"
    assert len(data["results"]) <= 5


def test_semantic_search_empty_query(client: FlaskClient):
    payload = {"query": "", "top_k": 5}
    response = client.post("/api/search", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "MISSING_QUERY"


def test_user_recommendations_known_and_cold_start(client: FlaskClient):
    # Known user
    response1 = client.get("/api/recommendations/user/USER_001?top_k=5")
    assert response1.status_code == 200
    data1 = response1.get_json()
    assert data1["success"] is True
    assert len(data1["recommendations"]) <= 5

    # Cold-start user
    response2 = client.get("/api/recommendations/user/UNKNOWN_USER_999?top_k=5")
    assert response2.status_code == 200
    data2 = response2.get_json()
    assert data2["success"] is True
    assert len(data2["recommendations"]) <= 5


def test_product_recommendations(client: FlaskClient):
    response = client.get("/api/recommendations/product/PROD_0101?top_k=5")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert len(data["recommendations"]) <= 5


def test_post_hybrid_recommendations(client: FlaskClient):
    payload = {
        "user_id": "USER_001",
        "product_id": "PROD_0101",
        "query": "comfortable headphones",
        "top_k": 3
    }
    response = client.post("/api/recommendations/hybrid", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert len(data["recommendations"]) <= 3
    rec = data["recommendations"][0]
    assert "final_hybrid_score" in rec
    assert "explanation" in rec


def test_post_hybrid_recommendations_invalid_weights(client: FlaskClient):
    payload = {
        "weights": {"content": 0.9, "collaborative": 0.9, "nlp": 0.1, "image": 0.1}
    }
    response = client.post("/api/recommendations/hybrid", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_WEIGHTS"


def test_recommendation_explanation_endpoint(client: FlaskClient):
    response = client.get("/api/recommendations/PROD_0102/explanation?reference_product_id=PROD_0101")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "explanation" in data
    assert "score_contributions" in data["explanation"]
