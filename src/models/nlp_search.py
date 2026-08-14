"""
NLP Semantic Search Engine Module

Uses SentenceTransformers (all-MiniLM-L6-v2) to map product titles, categories, and descriptions
into dense vector embeddings and search products using semantic cosine similarity.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NLPSemanticSearch:
    """
    Sentence-Transformer powered Semantic Search Engine.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.products_df: Optional[pd.DataFrame] = None
        self.product_embeddings: Optional[np.ndarray] = None
        self.product_id_to_idx: Dict[str, int] = {}

    def _init_model(self):
        """Lazy loads SentenceTransformer model."""
        if self.model is None:
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception as err:
                logger.error("Failed loading SentenceTransformer: %s. Using TF-IDF fallback vectorizer.", err)
                raise RuntimeError(f"SentenceTransformers model initialization failed: {err}") from err

    @staticmethod
    def _prepare_text(df: pd.DataFrame) -> List[str]:
        """Combines product title, category, and description into rich text representations."""
        names = df["product_name"].fillna("").astype(str)
        cats = df["category"].fillna("").astype(str)
        descs = df["description"].fillna("").astype(str)

        combined = (names + " Category: " + cats + ". Description: " + descs).tolist()
        return combined

    def fit(self, products_df: pd.DataFrame) -> "NLPSemanticSearch":
        """
        Encodes product catalog texts into dense 384D sentence embeddings.

        Args:
            products_df (pd.DataFrame): Processed products dataframe.

        Returns:
            NLPSemanticSearch: Fitted engine instance.
        """
        self._init_model()
        logger.info("Fitting NLPSemanticSearch on %d products...", len(products_df))
        self.products_df = products_df.copy().reset_index(drop=True)

        self.product_id_to_idx = {
            str(pid): idx for idx, pid in enumerate(self.products_df["product_id"])
        }

        combined_texts = self._prepare_text(self.products_df)
        
        # Encode texts into dense vectors (dimension 384)
        self.product_embeddings = self.model.encode(
            combined_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        logger.info(
            "NLPSemanticSearch fitted successfully. Embeddings shape: %s",
            self.product_embeddings.shape
        )
        return self

    def semantic_search(
        self, query: str, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Executes semantic search for a natural language text query.

        Args:
            query (str): User text query (e.g. "comfortable wireless headphones for gaming").
            top_k (int): Number of top results to return.

        Returns:
            pd.DataFrame: DataFrame containing product details and semantic similarity scores.
        """
        if self.product_embeddings is None or self.products_df is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        query_str = str(query).strip()
        if not query_str:
            logger.warning("Empty search query provided. Returning empty result.")
            return pd.DataFrame()

        self._init_model()

        # Encode query into 384D vector
        query_vec = self.model.encode(
            [query_str],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Calculate Cosine Similarity between query vector and product embeddings
        sim_scores = cosine_similarity(query_vec, self.product_embeddings).ravel()

        # Sort indices by similarity score descending
        sorted_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(sorted_indices, start=1):
            row = self.products_df.iloc[idx].to_dict()
            score = float(sim_scores[idx])

            results.append({
                "rank": rank,
                "product_id": row["product_id"],
                "product_name": row["product_name"],
                "category": row["category"],
                "price": row["price"],
                "rating": row["rating"],
                "similarity_score": round(score, 4),
                "query": query_str,
                "recommendation_method": f"NLP Semantic Search ({self.model_name})"
            })

        return pd.DataFrame(results)

    def save_model(self, filepath: Union[str, Path] = "models/nlp_search.joblib"):
        """Saves fitted NLP model embeddings & product dataframe to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save payload dictionary (avoid pickling PyTorch model object directly)
        payload = {
            "model_name": self.model_name,
            "products_df": self.products_df,
            "product_embeddings": self.product_embeddings,
            "product_id_to_idx": self.product_id_to_idx
        }
        joblib.dump(payload, filepath)
        logger.info("Saved NLPSemanticSearch artifact to %s", filepath)

    @classmethod
    def load_model(cls, filepath: Union[str, Path] = "models/nlp_search.joblib") -> "NLPSemanticSearch":
        """Loads fitted NLP embeddings artifact from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")

        payload = joblib.load(filepath)
        instance = cls(model_name=payload.get("model_name", "all-MiniLM-L6-v2"))
        instance.products_df = payload["products_df"]
        instance.product_embeddings = payload["product_embeddings"]
        instance.product_id_to_idx = payload["product_id_to_idx"]
        logger.info("Loaded NLPSemanticSearch artifact from %s", filepath)
        return instance


if __name__ == "__main__":
    from src.preprocessing.pipeline import DataPreprocessor
    clean_products = DataPreprocessor().run_pipeline()
    nlp_engine = NLPSemanticSearch().fit(clean_products)
    
    query = "comfortable wireless headphones for gaming"
    recs = nlp_engine.semantic_search(query, top_k=5)
    print(f"\nSemantic Search Results for Query: '{query}':")
    print(recs[["rank", "product_id", "product_name", "category", "similarity_score"]])
