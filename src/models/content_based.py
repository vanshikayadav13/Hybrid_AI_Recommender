"""
Content-Based Filtering Recommendation Engine

Uses text preprocessing, TF-IDF vectorization, and Cosine Similarity to recommend
products similar in features (title, category, description) to a given target product.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ContentBasedRecommender:
    """
    TF-IDF based Content Recommendation Engine.
    """

    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        self.products_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        self.product_id_to_idx: Dict[str, int] = {}
        self.idx_to_product_id: Dict[int, str] = {}
        self.feature_names: List[str] = []

    @staticmethod
    def _clean_text(text: str) -> str:
        """Helper to clean text string for vectorization."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        # Lowercase and remove punctuation/special characters
        text_clean = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", text_clean).strip()

    def _prepare_text_representation(self, df: pd.DataFrame) -> pd.Series:
        """
        Combines product name, category (weighted 2x for emphasis), and description.
        """
        names = df["product_name"].apply(self._clean_text) if "product_name" in df.columns else pd.Series([""] * len(df))
        categories = df["category"].apply(self._clean_text) if "category" in df.columns else pd.Series([""] * len(df))
        descriptions = df["description"].apply(self._clean_text) if "description" in df.columns else pd.Series([""] * len(df))

        # Weight category twice to boost category signal
        combined = names + " " + categories + " " + categories + " " + descriptions
        return combined

    def fit(self, products_df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Fits the TF-IDF vectorizer and calculates the pairwise cosine similarity matrix.

        Args:
            products_df (pd.DataFrame): Processed products dataframe.

        Returns:
            ContentBasedRecommender: Fitted recommender instance.
        """
        logger.info("Fitting ContentBasedRecommender on %d products...", len(products_df))
        self.products_df = products_df.copy().reset_index(drop=True)

        # Build index mappings
        self.product_id_to_idx = {
            str(pid): idx for idx, pid in enumerate(self.products_df["product_id"])
        }
        self.idx_to_product_id = {
            idx: str(pid) for idx, pid in enumerate(self.products_df["product_id"])
        }

        # Combine text fields
        combined_texts = self._prepare_text_representation(self.products_df)

        # Compute TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_texts)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        # Compute Cosine Similarity Matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        logger.info("ContentBasedRecommender fitted successfully. TF-IDF Matrix shape: %s", self.tfidf_matrix.shape)
        return self

    def explain_similarity(self, target_idx: int, cand_idx: int) -> str:
        """
        Generates a human-readable basic explanation of feature overlap between two products.
        """
        if self.tfidf_matrix is None or self.products_df is None:
            return "No explanation available."

        target_row = self.products_df.iloc[target_idx]
        cand_row = self.products_df.iloc[cand_idx]

        reasons = []

        # Check Category match
        if target_row["category"].lower() == cand_row["category"].lower():
            reasons.append(f"Same category ('{target_row['category']}')")
        else:
            reasons.append(f"Related categories ('{target_row['category']}' & '{cand_row['category']}')")

        # Find overlapping TF-IDF terms
        target_vector = self.tfidf_matrix[target_idx].toarray().ravel()
        cand_vector = self.tfidf_matrix[cand_idx].toarray().ravel()

        # Multiply elementwise to find features present in both
        overlap = target_vector * cand_vector
        top_overlap_indices = np.argsort(overlap)[::-1][:3]
        top_terms = [self.feature_names[i] for i in top_overlap_indices if overlap[i] > 0]

        if top_terms:
            reasons.append(f"Matching key terms: {', '.join(top_terms)}")

        return " | ".join(reasons)

    def recommend_similar_products(
        self, product_id: str, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Recommends top-K products most similar to the given product_id.

        Args:
            product_id (str): Target product identifier.
            top_k (int): Number of recommendations to return (default 10).

        Returns:
            pd.DataFrame: DataFrame containing recommended products, similarity scores, and explanations.

        Raises:
            ValueError: If product_id is not found in the fitted product catalog.
        """
        if self.similarity_matrix is None or self.products_df is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        pid_str = str(product_id).strip().upper()
        if pid_str not in self.product_id_to_idx:
            available_pids = list(self.product_id_to_idx.keys())[:5]
            raise ValueError(
                f"Product ID '{product_id}' not found in catalog. "
                f"Available example IDs: {available_pids}"
            )

        target_idx = self.product_id_to_idx[pid_str]

        # Get similarity scores for the target product
        sim_scores = self.similarity_matrix[target_idx].copy()

        # Sort indices by similarity descending
        sorted_indices = np.argsort(sim_scores)[::-1]

        # Exclude target product itself
        recommended_indices = [idx for idx in sorted_indices if idx != target_idx][:top_k]

        recommendations = []
        for rank, cand_idx in enumerate(recommended_indices, start=1):
            row = self.products_df.iloc[cand_idx].to_dict()
            score = float(sim_scores[cand_idx])
            explanation = self.explain_similarity(target_idx, cand_idx)

            row_dict = {
                "rank": rank,
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "price": row["price"],
                "rating": row["rating"],
                "similarity_score": round(score, 4),
                "explanation": explanation,
                "recommendation_method": "Content-Based Filtering (TF-IDF)"
            }
            recommendations.append(row_dict)

        return pd.DataFrame(recommendations)

    def save_model(self, filepath: Union[str, Path] = "models/content_based.joblib"):
        """Saves the fitted model artifact to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info("Saved ContentBasedRecommender artifact to %s", filepath)

    @classmethod
    def load_model(cls, filepath: Union[str, Path] = "models/content_based.joblib") -> "ContentBasedRecommender":
        """Loads fitted model artifact from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")
        model = joblib.load(filepath)
        logger.info("Loaded ContentBasedRecommender artifact from %s", filepath)
        return model


if __name__ == "__main__":
    from src.preprocessing.pipeline import DataPreprocessor
    clean_df = DataPreprocessor().run_pipeline()
    cb = ContentBasedRecommender().fit(clean_df)
    sample_pid = clean_df["product_id"].iloc[0]
    recs = cb.recommend_similar_products(sample_pid, top_k=5)
    print(f"\nRecommendations for Product {sample_pid}:")
    print(recs[["rank", "product_id", "product_name", "category", "similarity_score", "explanation"]])
