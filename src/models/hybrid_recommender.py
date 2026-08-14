"""
Hybrid Recommendation Engine Module

Ensembles four recommendation signals:
1. Content-Based Filtering (TF-IDF)
2. Collaborative Filtering (Item-Item)
3. NLP Semantic Search (SentenceTransformers 384D)
4. CNN Image Similarity (ResNet50 2048D)

Implements Candidate Generation, Min-Max Score Normalization, Configurable Weighted Fusion,
and Transparent Explainable AI (XAI) Attribution.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Set, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeRecommender
from src.models.cold_start import ColdStartHandler
from src.models.nlp_search import NLPSemanticSearch
from src.models.image_similarity import CNNImageSimilarity
from src.models.explainability import ExplainableAI
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default weights assigned across the 4 recommendation signals (Sum = 1.0)
DEFAULT_WEIGHTS = {
    "content": 0.25,
    "collaborative": 0.35,
    "nlp": 0.25,
    "image": 0.15
}


class HybridRecommender:
    """
    Ensemble Hybrid Recommendation Engine combining Content, Collaborative, NLP, and CNN Image signals.
    """

    def __init__(
        self,
        content_model: Optional[ContentBasedRecommender] = None,
        collaborative_model: Optional[CollaborativeRecommender] = None,
        nlp_model: Optional[NLPSemanticSearch] = None,
        image_model: Optional[CNNImageSimilarity] = None,
        products_df: Optional[pd.DataFrame] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.nlp_model = nlp_model
        self.image_model = image_model
        self.products_df = products_df
        self.weights = self.validate_weights(weights or DEFAULT_WEIGHTS)
        self.explainer = ExplainableAI()
        self.cold_start = ColdStartHandler(products_df) if products_df is not None else None

    @staticmethod
    def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validates weight configuration: must be non-negative and sum to 1.0 (within 1e-4 tolerance).

        Raises:
            ValueError: If weights are negative or do not sum to 1.0.
        """
        required_keys = {"content", "collaborative", "nlp", "image"}
        if not required_keys.issubset(weights.keys()):
            raise ValueError(f"Weights dict must contain all required keys: {required_keys}")

        for k, v in weights.items():
            if v < 0.0:
                raise ValueError(f"Weight for '{k}' cannot be negative. Got {v}")

        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-4:
            raise ValueError(f"Hybrid weights must sum to 1.0. Current sum: {weight_sum:.4f}")

        return weights.copy()

    @staticmethod
    def min_max_normalize(scores: pd.Series) -> pd.Series:
        """
        Applies Min-Max Normalization to scale raw scores into the [0.0, 1.0] interval.

        Formula:
            S_norm = (S - S_min) / (S_max - S_min)
        """
        s_min = float(scores.min())
        s_max = float(scores.max())

        if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
            # If all scores are equal or single candidate, return 1.0 if score > 0 else 0.0
            return scores.apply(lambda x: 1.0 if x > 0 else 0.0)

        denom = s_max - s_min
        if denom == 0:
            return scores.apply(lambda x: 1.0 if x > 0 else 0.0)

        norm = (scores - s_min) / denom
        return norm.clip(0.0, 1.0)

    def generate_candidates(
        self,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        query: Optional[str] = None,
        top_n: int = 25
    ) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Candidate Generation Stage:
        Queries available models to gather a pool of candidate products.

        Returns:
            Tuple[pd.DataFrame, Set[str]]: (candidate_df, interacted_items_set)
        """
        candidate_ids = set()
        interacted_items = set()

        # 1. Collaborative Candidates (if user_id provided)
        if user_id and self.collaborative_model:
            uid_str = str(user_id).strip().upper()
            if uid_str in self.collaborative_model.user_ids:
                user_ratings = self.collaborative_model.user_item_matrix.loc[uid_str]
                interacted_items = set(user_ratings[user_ratings > 0].index)
                cf_recs = self.collaborative_model.recommend_collaborative_products(uid_str, top_k=top_n)
                if not cf_recs.empty:
                    candidate_ids.update(cf_recs["product_id"].tolist())

        # 2. Content Candidates (if product_id provided)
        if product_id and self.content_model:
            pid_str = str(product_id).strip().upper()
            if pid_str in self.content_model.product_id_to_idx:
                cb_recs = self.content_model.recommend_similar_products(pid_str, top_k=top_n)
                if not cb_recs.empty:
                    candidate_ids.update(cb_recs["product_id"].tolist())

        # 3. NLP Candidates (if query provided)
        if query and self.nlp_model:
            nlp_recs = self.nlp_model.semantic_search(query, top_k=top_n)
            if not nlp_recs.empty:
                candidate_ids.update(nlp_recs["product_id"].tolist())

        # 4. Image Candidates (if product_id provided)
        if product_id and self.image_model:
            pid_str = str(product_id).strip().upper()
            if pid_str in self.image_model.product_id_to_idx:
                img_recs = self.image_model.find_similar_images(pid_str, top_k=top_n)
                if not img_recs.empty:
                    candidate_ids.update(img_recs["product_id"].tolist())

        # Exclude current reference product_id from candidate pool
        if product_id:
            candidate_ids.discard(str(product_id).strip().upper())

        # Exclude items user has already interacted with
        candidate_ids = candidate_ids - interacted_items

        # Fallback if candidates pool is empty (e.g. cold start user / empty query)
        if not candidate_ids and self.products_df is not None:
            pop_recs = self.cold_start.recommend_popular_products(top_k=top_n)
            candidate_ids.update(pop_recs["product_id"].tolist())
            if product_id:
                candidate_ids.discard(str(product_id).strip().upper())

        # Filter products dataframe to candidate pool
        cand_df = self.products_df[self.products_df["product_id"].isin(candidate_ids)].copy().reset_index(drop=True)
        return cand_df, interacted_items

    def hybrid_recommend(
        self,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Executes unified hybrid recommendation pipeline across 4 signals.

        Args:
            user_id (Optional[str]): Target user ID.
            product_id (Optional[str]): Reference product ID.
            query (Optional[str]): Text search query.
            top_k (int): Number of hybrid recommendations to return.
            weights (Optional[Dict[str, float]]): Custom weight dictionary override.

        Returns:
            pd.DataFrame: Ranked hybrid recommendations dataframe with XAI explanation breakdown.
        """
        active_weights = self.validate_weights(weights) if weights else self.weights.copy()

        # Step 1: Candidate Generation
        cand_df, _ = self.generate_candidates(user_id=user_id, product_id=product_id, query=query, top_n=30)
        if cand_df.empty:
            logger.warning("No candidate products generated.")
            return pd.DataFrame()

        cand_pids = cand_df["product_id"].tolist()

        # Step 2: Compute Raw Scores per model
        cand_df["raw_content_score"] = 0.0
        cand_df["raw_collaborative_score"] = 0.0
        cand_df["raw_nlp_score"] = 0.0
        cand_df["raw_image_score"] = 0.0

        # Raw Content Scores
        if product_id and self.content_model:
            pid_str = str(product_id).strip().upper()
            if pid_str in self.content_model.product_id_to_idx:
                target_idx = self.content_model.product_id_to_idx[pid_str]
                sims = self.content_model.similarity_matrix[target_idx]
                for i, c_pid in enumerate(cand_pids):
                    if c_pid in self.content_model.product_id_to_idx:
                        c_idx = self.content_model.product_id_to_idx[c_pid]
                        cand_df.loc[i, "raw_content_score"] = float(sims[c_idx])

        # Raw Collaborative Scores
        if user_id and self.collaborative_model:
            uid_str = str(user_id).strip().upper()
            if uid_str in self.collaborative_model.user_ids:
                for i, c_pid in enumerate(cand_pids):
                    cand_df.loc[i, "raw_collaborative_score"] = self.collaborative_model.predict_rating(uid_str, c_pid)

        # Raw NLP Scores
        if query and self.nlp_model:
            query_vec = self.nlp_model.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            nlp_sims = cosine_similarity(query_vec, self.nlp_model.product_embeddings).ravel()
            for i, c_pid in enumerate(cand_pids):
                if c_pid in self.nlp_model.product_id_to_idx:
                    c_idx = self.nlp_model.product_id_to_idx[c_pid]
                    cand_df.loc[i, "raw_nlp_score"] = float(nlp_sims[c_idx])

        # Raw Image Scores
        if product_id and self.image_model:
            pid_str = str(product_id).strip().upper()
            if pid_str in self.image_model.product_id_to_idx:
                target_idx = self.image_model.product_id_to_idx[pid_str]
                target_vec = self.image_model.image_features[target_idx].reshape(1, -1)
                img_sims = cosine_similarity(target_vec, self.image_model.image_features).ravel()
                for i, c_pid in enumerate(cand_pids):
                    if c_pid in self.image_model.product_id_to_idx:
                        c_idx = self.image_model.product_id_to_idx[c_pid]
                        cand_df.loc[i, "raw_image_score"] = float(img_sims[c_idx])

        # Step 3: Min-Max Normalization per signal
        cand_df["norm_content_score"] = self.min_max_normalize(cand_df["raw_content_score"])
        cand_df["norm_collaborative_score"] = self.min_max_normalize(cand_df["raw_collaborative_score"])
        cand_df["norm_nlp_score"] = self.min_max_normalize(cand_df["raw_nlp_score"])
        cand_df["norm_image_score"] = self.min_max_normalize(cand_df["raw_image_score"])

        # Determine active signals and dynamically re-normalize weights if necessary
        active_signals = {
            "content": bool(product_id and self.content_model),
            "collaborative": bool(user_id and self.collaborative_model and str(user_id).strip().upper() in self.collaborative_model.user_ids),
            "nlp": bool(query and self.nlp_model),
            "image": bool(product_id and self.image_model)
        }

        active_weight_sum = sum(active_weights[sig] for sig, active in active_signals.items() if active)
        if active_weight_sum > 0:
            effective_weights = {
                sig: (active_weights[sig] / active_weight_sum if active_signals[sig] else 0.0)
                for sig in active_weights
            }
        else:
            effective_weights = active_weights.copy()

        # Step 4: Weighted Fusion
        cand_df["final_hybrid_score"] = (
            effective_weights["content"] * cand_df["norm_content_score"] +
            effective_weights["collaborative"] * cand_df["norm_collaborative_score"] +
            effective_weights["nlp"] * cand_df["norm_nlp_score"] +
            effective_weights["image"] * cand_df["norm_image_score"]
        ).round(4)

        # Sort candidates by final_hybrid_score descending
        ranked_df = cand_df.sort_values(by="final_hybrid_score", ascending=False).head(top_k).reset_index(drop=True)

        # Step 5: Attach Explainable AI (XAI) Attribution
        ref_product_name = None
        if product_id and self.products_df is not None:
            match = self.products_df[self.products_df["product_id"] == str(product_id).strip().upper()]
            if not match.empty:
                ref_product_name = match.iloc[0]["product_name"]

        enriched_results = []
        for rank, row in ranked_df.iterrows():
            row_dict = row.to_dict()
            row_dict["rank"] = rank + 1
            explained_row = self.explainer.explain_recommendation(
                row_dict,
                effective_weights,
                query=query,
                product_context=ref_product_name
            )
            enriched_results.append(explained_row)

        output_df = pd.DataFrame(enriched_results)

        cols_order = [
            "rank", "product_id", "product_name", "category", "price", "rating",
            "final_hybrid_score", "content_contribution", "collaborative_contribution",
            "nlp_contribution", "image_contribution", "explanation"
        ]
        return output_df[[c for c in cols_order if c in output_df.columns]]

    def save_model(self, filepath: Union[str, Path] = "models/hybrid_recommender.joblib"):
        """Saves fitted hybrid config and weights to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"weights": self.weights}, filepath)
        logger.info("Saved HybridRecommender configuration to %s", filepath)

    @classmethod
    def load_model(
        cls,
        filepath: Union[str, Path] = "models/hybrid_recommender.joblib",
        content_model: Optional[ContentBasedRecommender] = None,
        collaborative_model: Optional[CollaborativeRecommender] = None,
        nlp_model: Optional[NLPSemanticSearch] = None,
        image_model: Optional[CNNImageSimilarity] = None,
        products_df: Optional[pd.DataFrame] = None
    ) -> "HybridRecommender":
        """Loads hybrid configuration artifact from disk."""
        filepath = Path(filepath)
        weights = DEFAULT_WEIGHTS
        if filepath.exists():
            data = joblib.load(filepath)
            weights = data.get("weights", DEFAULT_WEIGHTS)
            logger.info("Loaded HybridRecommender configuration from %s", filepath)

        return cls(
            content_model=content_model,
            collaborative_model=collaborative_model,
            nlp_model=nlp_model,
            image_model=image_model,
            products_df=products_df,
            weights=weights
        )


if __name__ == "__main__":
    from src.preprocessing.pipeline import DataPreprocessor
    clean_products = DataPreprocessor().run_pipeline()
    
    cb = ContentBasedRecommender.load_model("models/content_based.joblib")
    cf = CollaborativeRecommender.load_model("models/collaborative.joblib")
    nlp = NLPSemanticSearch.load_model("models/nlp_search.joblib")
    img = CNNImageSimilarity.load_model("models/image_features.joblib")
    
    hybrid = HybridRecommender(
        content_model=cb,
        collaborative_model=cf,
        nlp_model=nlp,
        image_model=img,
        products_df=clean_products
    )
    
    sample_pid = clean_products["product_id"].iloc[0]
    sample_uid = cf.user_ids[0]
    
    recs = hybrid.hybrid_recommend(
        user_id=sample_uid,
        product_id=sample_pid,
        query="comfortable headphones for audio",
        top_k=5
    )
    print("\nHybrid Recommendations:")
    print(recs)
