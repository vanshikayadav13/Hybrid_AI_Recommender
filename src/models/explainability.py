"""
Transparent Explainable AI (XAI) Engine Module

Provides score-based attribution and human-readable reasoning for hybrid recommendations.
Calculates exact mathematical score contribution per recommendation signal:
    Contribution_j = Weight_j * Normalized_Score_j
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExplainableAI:
    """
    Transparent Score-Attribution Explainable AI Engine for Hybrid Recommendations.
    """

    @staticmethod
    def calculate_score_contributions(
        normalized_scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculates exact mathematical score contribution for each recommendation signal.

        Formula:
            Contribution_j = Weight_j * Normalized_Score_j

        Args:
            normalized_scores (Dict[str, float]): Map of signal_name -> normalized score in [0.0, 1.0].
            weights (Dict[str, float]): Map of signal_name -> weight in [0.0, 1.0].

        Returns:
            Dict[str, float]: Map of signal_name -> calculated contribution score.
        """
        contributions = {}
        for signal, norm_score in normalized_scores.items():
            w = weights.get(signal, 0.0)
            contributions[signal] = round(float(w * norm_score), 4)
        return contributions

    @staticmethod
    def generate_explanation_text(
        contributions: Dict[str, float],
        query: str = None,
        product_context: str = None,
        threshold: float = 0.05
    ) -> str:
        """
        Generates human-readable explanation bullet points based on dominant signal contributions.

        Args:
            contributions (Dict[str, float]): Signal contribution scores.
            query (str): User text query (if applicable).
            product_context (str): Reference product name (if applicable).
            threshold (float): Minimum contribution required to trigger an explanation line.

        Returns:
            str: Human-readable explanation string.
        """
        reasons = []

        content_contrib = contributions.get("content", 0.0)
        collab_contrib = contributions.get("collaborative", 0.0)
        nlp_contrib = contributions.get("nlp", 0.0)
        image_contrib = contributions.get("image", 0.0)

        if nlp_contrib >= threshold:
            if query:
                reasons.append(f"Strong semantic match with your query '{query}' (Score: +{nlp_contrib:.2f})")
            else:
                reasons.append(f"High semantic text similarity (Score: +{nlp_contrib:.2f})")

        if collab_contrib >= threshold:
            reasons.append(f"Preferred by users with similar shopping preferences (Score: +{collab_contrib:.2f})")

        if content_contrib >= threshold:
            if product_context:
                reasons.append(f"Category & description match reference item '{product_context}' (Score: +{content_contrib:.2f})")
            else:
                reasons.append(f"Product title & description match your interest (Score: +{content_contrib:.2f})")

        if image_contrib >= threshold:
            reasons.append(f"Visually resembles reference product image (Score: +{image_contrib:.2f})")

        if not reasons:
            reasons.append("Recommended based on catalog popularity and category relevance")

        return " | ".join(reasons)

    def explain_recommendation(
        self,
        row: Dict[str, Any],
        weights: Dict[str, float],
        query: str = None,
        product_context: str = None
    ) -> Dict[str, Any]:
        """
        Enriches a recommendation row dictionary with exact contribution breakdown and explanation text.
        """
        norm_scores = {
            "content": float(row.get("norm_content_score", 0.0)),
            "collaborative": float(row.get("norm_collaborative_score", 0.0)),
            "nlp": float(row.get("norm_nlp_score", 0.0)),
            "image": float(row.get("norm_image_score", 0.0))
        }

        contributions = self.calculate_score_contributions(norm_scores, weights)
        explanation_text = self.generate_explanation_text(contributions, query=query, product_context=product_context)

        row_copy = row.copy()
        row_copy["content_contribution"] = contributions["content"]
        row_copy["collaborative_contribution"] = contributions["collaborative"]
        row_copy["nlp_contribution"] = contributions["nlp"]
        row_copy["image_contribution"] = contributions["image"]
        row_copy["explanation"] = explanation_text
        row_copy["xai_limitation_notice"] = (
            "Transparent score attribution model based on weighted signal combination. "
            "Reflects model score breakdown; does not represent causal user intent."
        )
        return row_copy
