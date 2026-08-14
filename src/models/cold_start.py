"""
Cold Start Handler & Unified Recommendation Engine Facade

Provides fallback strategies for:
1. New Users with no interaction history -> Popular / Top-Rated items
2. New Products with no interaction history -> Content-Based similarity matching
3. Unified Recommendation Engine routing queries appropriately.
"""

from typing import Dict, List, Optional, Union
import pandas as pd

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeRecommender
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ColdStartHandler:
    """
    Handles recommendation fallbacks when user or product interaction history is missing.
    """

    def __init__(self, products_df: pd.DataFrame):
        self.products_df = products_df.copy()

    def recommend_popular_products(
        self, top_k: int = 10, category: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Recommends highest-rated, popular products for a new user with zero interaction history.

        Args:
            top_k (int): Number of items to recommend.
            category (Optional[str]): Optional category filter.

        Returns:
            pd.DataFrame: Popular products dataset.
        """
        df = self.products_df.copy()

        if category:
            filtered = df[df["category"].str.lower() == category.lower()]
            if not filtered.empty:
                df = filtered

        # Sort by rating descending, then price descending
        popular_df = df.sort_values(by=["rating", "price"], ascending=[False, False]).head(top_k).copy()
        
        results = []
        for rank, (_, row) in enumerate(popular_df.iterrows(), start=1):
            results.append({
                "rank": rank,
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "price": row["price"],
                "rating": row["rating"],
                "predicted_score": round(float(row["rating"]), 2),
                "explanation": "Top-rated popular product across catalog (Cold-Start fallback)",
                "recommendation_method": "Cold-Start Popularity Strategy"
            })

        return pd.DataFrame(results)

    def recommend_for_new_product(
        self, content_recommender: ContentBasedRecommender, new_product_id: str, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Uses content similarity to recommend similar existing items for a new product.
        """
        return content_recommender.recommend_similar_products(new_product_id, top_k=top_k)


class RecommendationEngine:
    """
    Unified Facade wrapping Content-Based, Collaborative, and Cold-Start Recommenders.
    """

    def __init__(
        self,
        content_model: Optional[ContentBasedRecommender] = None,
        collaborative_model: Optional[CollaborativeRecommender] = None,
        products_df: Optional[pd.DataFrame] = None
    ):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.products_df = products_df
        self.cold_start = ColdStartHandler(products_df) if products_df is not None else None

    def recommend_for_product(self, product_id: str, top_k: int = 10) -> pd.DataFrame:
        """Content-based item recommendation facade."""
        if self.content_model is None:
            raise RuntimeError("Content-Based model is not initialized.")
        return self.content_model.recommend_similar_products(product_id, top_k=top_k)

    def recommend_for_user(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        """
        Collaborative recommendation facade with Cold-Start fallback for new users.
        """
        if self.collaborative_model is None:
            raise RuntimeError("Collaborative model is not initialized.")

        # Check if user exists in collaborative model
        uid_str = str(user_id).strip().upper()
        if uid_str not in self.collaborative_model.user_ids:
            logger.info("User '%s' is unknown. Applying Cold-Start popularity strategy.", user_id)
            if self.cold_start:
                return self.cold_start.recommend_popular_products(top_k=top_k)
            return pd.DataFrame()

        return self.collaborative_model.recommend_collaborative_products(user_id, top_k=top_k)
