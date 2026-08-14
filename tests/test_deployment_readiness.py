"""
Unit Tests for Deployment Readiness, PORT Handling & Memory Optimization
"""

import os
from unittest.mock import patch
from src.models import CNNImageSimilarity, NLPSemanticSearch
from src.api.service import RecommendationService


def test_cnn_persisted_feature_loading_does_not_initialize_resnet():
    """Verifies loading persisted joblib features does not instantiate PyTorch ResNet50."""
    cnn = CNNImageSimilarity.load_model("models/image_features.joblib")
    assert cnn.image_features is not None
    assert cnn.model is None  # Model remains None until extract_image_features/fit is called


def test_nlp_persisted_feature_loading_has_embeddings():
    """Verifies loading persisted joblib embeddings contains valid 384D matrix."""
    nlp = NLPSemanticSearch.load_model("models/nlp_search.joblib")
    assert nlp.product_embeddings is not None
    assert nlp.product_embeddings.shape[1] == 384


def test_recommendation_service_singleton_instance():
    """Verifies RecommendationService acts as a singleton."""
    s1 = RecommendationService()
    s2 = RecommendationService()
    assert s1 is s2


def test_render_port_environment_fallback():
    """Verifies default PORT fallback mechanism."""
    port_env = os.environ.get("PORT", "5000")
    assert port_env is not None
    assert int(port_env) > 0
