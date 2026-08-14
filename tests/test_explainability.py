"""
Unit Tests for Explainable AI (XAI) Module
"""

from src.models.explainability import ExplainableAI


def test_score_contributions_calculation():
    norm_scores = {
        "content": 0.80,
        "collaborative": 0.60,
        "nlp": 0.90,
        "image": 0.50
    }
    weights = {
        "content": 0.25,
        "collaborative": 0.35,
        "nlp": 0.25,
        "image": 0.15
    }

    # Expected:
    # content: 0.25 * 0.80 = 0.20
    # collaborative: 0.35 * 0.60 = 0.21
    # nlp: 0.25 * 0.90 = 0.225
    # image: 0.15 * 0.50 = 0.075
    contributions = ExplainableAI.calculate_score_contributions(norm_scores, weights)

    assert contributions["content"] == 0.20
    assert contributions["collaborative"] == 0.21
    assert contributions["nlp"] == 0.225
    assert contributions["image"] == 0.075

    # Sum of contributions equals final score
    total_score = sum(contributions.values())
    assert round(total_score, 4) == 0.71


def test_generate_explanation_text():
    contributions = {
        "content": 0.20,
        "collaborative": 0.21,
        "nlp": 0.225,
        "image": 0.02  # Below threshold
    }

    text = ExplainableAI.generate_explanation_text(
        contributions, query="wireless gaming headset", product_context="Headphones A"
    )

    assert "wireless gaming headset" in text
    assert "Headphones A" in text
    assert "users with similar shopping preferences" in text
    # Image contribution is below threshold 0.05, so it should not appear
    assert "Visually resembles" not in text


def test_explain_recommendation_row():
    row = {
        "product_id": "P101",
        "product_name": "Wireless Headphones",
        "norm_content_score": 0.8,
        "norm_collaborative_score": 0.7,
        "norm_nlp_score": 0.9,
        "norm_image_score": 0.4
    }
    weights = {"content": 0.25, "collaborative": 0.35, "nlp": 0.25, "image": 0.15}

    explainer = ExplainableAI()
    explained = explainer.explain_recommendation(row, weights, query="audio headphones")

    assert "content_contribution" in explained
    assert "collaborative_contribution" in explained
    assert "nlp_contribution" in explained
    assert "image_contribution" in explained
    assert "explanation" in explained
    assert "xai_limitation_notice" in explained
