"""
Item-Item Collaborative Filtering Recommendation Engine

Pivots user interaction logs into a User-Item rating matrix, computes pairwise item similarity,
and predicts user preferences for unrated products based on past item ratings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CollaborativeRecommender:
    """
    Item-Item Collaborative Filtering Engine.
    """

    def __init__(self, min_interactions: int = 1):
        self.min_interactions = min_interactions
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.item_similarity_matrix: Optional[np.ndarray] = None
        self.item_similarity_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.user_ids: List[str] = []
        self.product_ids: List[str] = []

    def fit(
        self,
        interactions_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None
    ) -> "CollaborativeRecommender":
        """
        Builds the User-Item rating matrix and calculates Item-Item Cosine Similarity.

        Args:
            interactions_df (pd.DataFrame): DataFrame containing columns [user_id, product_id, user_rating].
            products_df (Optional[pd.DataFrame]): Catalog metadata for product enrichment.

        Returns:
            CollaborativeRecommender: Fitted recommender instance.
        """
        logger.info("Fitting CollaborativeRecommender on %d interaction logs...", len(interactions_df))
        
        if products_df is not None:
            self.products_df = products_df.copy()

        # Filter required columns
        df = interactions_df[["user_id", "product_id", "user_rating"]].dropna().copy()
        df["user_id"] = df["user_id"].astype(str).str.strip().str.upper()
        df["product_id"] = df["product_id"].astype(str).str.strip().str.upper()

        # Build User-Item Interaction Matrix (Users as rows, Products as columns)
        self.user_item_matrix = df.pivot_table(
            index="user_id",
            columns="product_id",
            values="user_rating",
            aggfunc="mean"
        ).fillna(0.0)

        self.user_ids = self.user_item_matrix.index.tolist()
        self.product_ids = self.user_item_matrix.columns.tolist()

        # Compute Item-Item Similarity Matrix (Transpose to get Items as rows)
        item_vectors = self.user_item_matrix.T.values
        self.item_similarity_matrix = cosine_similarity(item_vectors)

        # Wrap in DataFrame for easy lookup
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.product_ids,
            columns=self.product_ids
        )

        logger.info(
            "CollaborativeRecommender fitted successfully. Matrix shape: Users=%d, Products=%d",
            len(self.user_ids), len(self.product_ids)
        )
        return self

    def predict_rating(self, user_id: str, product_id: str) -> float:
        """
        Predicts user rating for a target product using weighted sum of item similarities.

        Formula:
            r_pred(u, i) = sum(S(i, j) * r(u, j)) / sum(|S(i, j)|) for j in User's rated items.
        """
        if self.user_item_matrix is None or self.item_similarity_df is None:
            raise RuntimeError("Model is not fitted yet.")

        user_id_str = str(user_id).strip().upper()
        product_id_str = str(product_id).strip().upper()

        if user_id_str not in self.user_item_matrix.index or product_id_str not in self.item_similarity_df.columns:
            return 0.0

        # Get ratings given by user
        user_ratings = self.user_item_matrix.loc[user_id_str]
        rated_items = user_ratings[user_ratings > 0].index

        if len(rated_items) == 0:
            return 0.0

        # Get similarity between target product and all rated products
        sim_scores = self.item_similarity_df.loc[product_id_str, rated_items].values
        ratings = user_ratings[rated_items].values

        sim_sum = np.sum(np.abs(sim_scores))
        if sim_sum == 0:
            return 0.0

        predicted_rating = float(np.sum(sim_scores * ratings) / sim_sum)
        return round(predicted_rating, 4)

    def recommend_collaborative_products(
        self, user_id: str, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Recommends top-K products for a user based on Item-Item Collaborative Filtering.

        Args:
            user_id (str): Target user identifier.
            top_k (int): Number of products to recommend.

        Returns:
            pd.DataFrame: DataFrame containing recommended products and predicted preference scores.
        """
        if self.user_item_matrix is None or self.item_similarity_df is None:
            raise RuntimeError("Model is not fitted yet.")

        uid_str = str(user_id).strip().upper()
        if uid_str not in self.user_item_matrix.index:
            logger.warning("User ID '%s' not found in interaction matrix (Cold Start user).", user_id)
            return pd.DataFrame()

        # Identify items user has already interacted with
        user_ratings = self.user_item_matrix.loc[uid_str]
        interacted_items = set(user_ratings[user_ratings > 0].index)

        # Candidate items are items user hasn't interacted with yet
        candidate_items = [p for p in self.product_ids if p not in interacted_items]

        scores = []
        for pid in candidate_items:
            pred_score = self.predict_rating(uid_str, pid)
            if pred_score > 0:
                scores.append((pid, pred_score))

        # Sort candidate items by predicted score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]

        recommendations = []
        for rank, (pid, score) in enumerate(top_scores, start=1):
            prod_name = pid
            cat = "Unknown"
            price = 0.0
            rating = 0.0

            if self.products_df is not None:
                match = self.products_df[self.products_df["product_id"] == pid]
                if not match.empty:
                    row = match.iloc[0]
                    prod_name = row["product_name"]
                    cat = row["category"]
                    price = row["price"]
                    rating = row["rating"]

            recommendations.append({
                "rank": rank,
                "product_id": pid,
                "product_name": prod_name,
                "category": cat,
                "price": price,
                "rating": rating,
                "predicted_score": score,
                "explanation": f"Recommended based on past item ratings by user {uid_str}",
                "recommendation_method": "Collaborative Filtering (Item-Item)"
            })

        return pd.DataFrame(recommendations)

    def save_model(self, filepath: Union[str, Path] = "models/collaborative.joblib"):
        """Saves fitted collaborative filtering model artifact to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved CollaborativeRecommender artifact to %s", filepath)

    @classmethod
    def load_model(cls, filepath: Union[str, Path] = "models/collaborative.joblib") -> "CollaborativeRecommender":
        """Loads fitted collaborative filtering model artifact from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        model = joblib.load(filepath)
        logger.info("Loaded CollaborativeRecommender artifact from %s", filepath)
        return model


if __name__ == "__main__":
    from src.preprocessing.pipeline import DataPreprocessor
    clean_products = DataPreprocessor().run_pipeline()
    inter_path = Path("data/raw/user_interactions_raw.csv")
    if inter_path.exists():
        inter_df = pd.read_csv(inter_path)
        cf = CollaborativeRecommender().fit(inter_df, clean_products)
        sample_user = cf.user_ids[0]
        recs = cf.recommend_collaborative_products(sample_user, top_k=5)
        print(f"\nCollaborative Recommendations for User {sample_user}:")
        print(recs)
