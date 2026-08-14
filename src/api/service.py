"""
Singleton ML Service Manager for Flask Backend

Loads serialized ML models into memory once at server startup to ensure high performance
and low latency during API request handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from src.preprocessing.pipeline import DataPreprocessor
from src.data.loader import DataLoader
from src.models import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    NLPSemanticSearch,
    CNNImageSimilarity,
    HybridRecommender,
    ExplainableAI,
    DEFAULT_WEIGHTS
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecommendationService:
    """
    Singleton ML Service Manager class.
    """
    _instance: Optional["RecommendationService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecommendationService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(
        self,
        data_dir: Path = Path("data"),
        models_dir: Path = Path("models")
    ):
        """Initializes and loads all ML models and dataset artifacts into memory."""
        if self._initialized:
            logger.info("RecommendationService is already initialized.")
            return

        logger.info("Initializing RecommendationService and loading ML model artifacts...")
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)

        # 1. Load Products Catalog
        processed_path = self.data_dir / "processed" / "clean_products.csv"
        if processed_path.exists():
            self.products_df = pd.read_csv(processed_path)
        else:
            logger.warning("Clean products file not found. Running preprocessor pipeline...")
            self.products_df = DataPreprocessor().run_pipeline()

        # 2. Load User Interactions
        raw_interactions = self.data_dir / "raw" / "user_interactions_raw.csv"
        if raw_interactions.exists():
            self.interactions_df = pd.read_csv(raw_interactions)
        else:
            logger.warning("User interactions file not found. Loading synthetic dataset...")
            self.interactions_df = DataLoader().load_raw_csv()

        # 3. Load ML Models
        try:
            self.cb_model = ContentBasedRecommender.load_model(self.models_dir / "content_based.joblib")
        except Exception as e:
            logger.warning(f"Failed to load ContentBasedRecommender: {e}. Fitting fresh instance...")
            self.cb_model = ContentBasedRecommender().fit(self.products_df)

        try:
            self.cf_model = CollaborativeRecommender.load_model(self.models_dir / "collaborative.joblib")
        except Exception as e:
            logger.warning(f"Failed to load CollaborativeRecommender: {e}. Fitting fresh instance...")
            self.cf_model = CollaborativeRecommender().fit(self.interactions_df, self.products_df)

        try:
            self.nlp_model = NLPSemanticSearch.load_model(self.models_dir / "nlp_search.joblib")
        except Exception as e:
            logger.warning(f"Failed to load NLPSemanticSearch: {e}. Fitting fresh instance...")
            self.nlp_model = NLPSemanticSearch().fit(self.products_df)

        try:
            self.img_model = CNNImageSimilarity.load_model(self.models_dir / "image_features.joblib")
        except Exception as e:
            logger.warning(f"Failed to load CNNImageSimilarity: {e}. Fitting fresh instance...")
            self.img_model = CNNImageSimilarity().fit(self.products_df)

        # 4. Initialize Unified Hybrid Engine
        try:
            self.hybrid_engine = HybridRecommender.load_model(
                self.models_dir / "hybrid_recommender.joblib",
                content_model=self.cb_model,
                collaborative_model=self.cf_model,
                nlp_model=self.nlp_model,
                image_model=self.img_model,
                products_df=self.products_df
            )
        except Exception as e:
            logger.warning(f"Failed to load HybridRecommender artifact: {e}. Initializing default...")
            self.hybrid_engine = HybridRecommender(
                content_model=self.cb_model,
                collaborative_model=self.cf_model,
                nlp_model=self.nlp_model,
                image_model=self.img_model,
                products_df=self.products_df,
                weights=DEFAULT_WEIGHTS
            )

        self.explainer = ExplainableAI()
        self._initialized = True
        logger.info("RecommendationService initialization complete. All ML models ready.")

    def get_products(
        self,
        category: Optional[str] = None,
        search_term: Optional[str] = None,
        page: int = 1,
        limit: int = 12
    ) -> Dict[str, Any]:
        """Retrieves paginated product catalog with optional filtering."""
        df = self.products_df.copy()

        if category and category.lower() != "all":
            df = df[df["category"].str.lower() == category.lower()]

        if search_term and search_term.strip():
            term = search_term.strip().lower()
            df = df[
                df["product_name"].str.lower().str.contains(term) |
                df["description"].str.lower().str.contains(term)
            ]

        total_items = len(df)
        total_pages = max(1, (total_items + limit - 1) // limit)
        page = max(1, min(page, total_pages))

        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_items = df.iloc[start_idx:end_idx].to_dict(orient="records")

        return {
            "products": page_items,
            "pagination": {
                "total_items": total_items,
                "page": page,
                "limit": limit,
                "total_pages": total_pages
            }
        }

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves single product metadata by product_id."""
        pid_clean = str(product_id).strip().upper()
        match = self.products_df[self.products_df["product_id"] == pid_clean]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def search_semantic(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Executes SentenceTransformer semantic vector search."""
        if not query or not query.strip():
            return []
        recs = self.nlp_model.semantic_search(query.strip(), top_k=top_k)
        return recs.to_dict(orient="records") if not recs.empty else []

    def get_recommendations_hybrid(
        self,
        user_id: Optional[str] = None,
        product_id: Optional[str] = None,
        query: Optional[str] = None,
        top_k: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Executes unified hybrid recommendation pipeline."""
        recs_df = self.hybrid_engine.hybrid_recommend(
            user_id=user_id,
            product_id=product_id,
            query=query,
            top_k=top_k,
            weights=weights
        )
        if recs_df.empty:
            return []
        return recs_df.to_dict(orient="records")


# Helper function to get service instance
def get_service() -> RecommendationService:
    service = RecommendationService()
    if not service._initialized:
        service.initialize()
    return service
