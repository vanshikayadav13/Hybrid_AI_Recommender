"""
CNN Image Similarity Engine Module

Uses a pretrained ResNet50 Convolutional Neural Network (with fully-connected classification
layer truncated) to extract 2048-dimensional feature embeddings from product images
and compute visual cosine similarity.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from torchvision import models, transforms

from src.data.image_utils import safe_load_image, ensure_catalog_images
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CNNImageSimilarity:
    """
    ResNet50 CNN Feature Extractor & Visual Similarity Search Engine.
    """

    def __init__(self, use_gpu: bool = False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.image_features: Optional[np.ndarray] = None
        self.product_id_to_idx: Dict[str, int] = {}
        self.image_paths: Dict[str, Path] = {}

        # Standard ResNet / ImageNet preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _init_resnet(self):
        """Loads pretrained ResNet50 and truncates final FC layer to Identity."""
        if self.model is None:
            logger.info("Initializing pretrained ResNet50 Feature Extractor on %s...", self.device)
            try:
                # Try modern torchvision weights API first, fallback to pretrained=True
                try:
                    weights = models.ResNet50_Weights.DEFAULT
                    model = models.resnet50(weights=weights)
                except AttributeError:
                    model = models.resnet50(pretrained=True)

                # Truncate classification layer to return 2048-dim feature vector
                model.fc = nn.Identity()
                model.eval()
                model.to(self.device)
                self.model = model
                logger.info("ResNet50 Feature Extractor initialized successfully (Output Dim: 2048).")
            except Exception as err:
                logger.error("Failed initializing ResNet50 model: %s", err)
                raise RuntimeError(f"ResNet50 initialization failed: {err}") from err

    def extract_image_features(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Extracts 2048D feature vector from a single image file.

        Args:
            image_path (Union[str, Path]): Path to image.

        Returns:
            np.ndarray: 2048-dimensional feature vector.
        """
        self._init_resnet()
        
        # Load and format image to RGB
        pil_img = safe_load_image(image_path)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor_img)
            # Normalize vector to unit length for fast cosine similarity
            norm_features = torch.nn.functional.normalize(features, p=2, dim=1)
            vector = norm_features.squeeze(0).cpu().numpy()

        return vector

    def fit(
        self,
        products_df: pd.DataFrame,
        image_dir: Path = Path("data/images")
    ) -> "CNNImageSimilarity":
        """
        Extracts and caches 2048D feature vectors for all catalog product images.

        Args:
            products_df (pd.DataFrame): Products catalog dataset.
            image_dir (Path): Local image storage directory.

        Returns:
            CNNImageSimilarity: Fitted engine instance.
        """
        self._init_resnet()
        logger.info("Fitting CNNImageSimilarity on %d product images...", len(products_df))
        self.products_df = products_df.copy().reset_index(drop=True)

        # Ensure local images exist (generate synthetic image fallbacks if absent)
        self.image_paths = ensure_catalog_images(self.products_df, image_dir=image_dir)

        self.product_id_to_idx = {
            str(pid): idx for idx, pid in enumerate(self.products_df["product_id"])
        }

        features_list = []
        for idx, row in self.products_df.iterrows():
            pid = str(row["product_id"]).strip().upper()
            img_path = self.image_paths.get(pid)

            try:
                vec = self.extract_image_features(img_path)
            except Exception as err:
                logger.warning("Error extracting features for product %s at %s: %s. Using zero vector.", pid, img_path, err)
                vec = np.zeros(2048, dtype=np.float32)

            features_list.append(vec)

        self.image_features = np.vstack(features_list)
        logger.info(
            "CNNImageSimilarity fitted successfully. Feature matrix shape: %s",
            self.image_features.shape
        )
        return self

    def find_similar_images(
        self, product_id: str, top_k: int = 10
    ) -> pd.DataFrame:
        """
        Finds visually similar products based on CNN feature vector cosine similarity.

        Args:
            product_id (str): Target product identifier.
            top_k (int): Number of visually similar products to return.

        Returns:
            pd.DataFrame: DataFrame containing visually similar products and similarity scores.
        """
        if self.image_features is None or self.products_df is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        pid_str = str(product_id).strip().upper()
        if pid_str not in self.product_id_to_idx:
            available_pids = list(self.product_id_to_idx.keys())[:5]
            raise ValueError(
                f"Product ID '{product_id}' not found in image feature index. "
                f"Available example IDs: {available_pids}"
            )

        target_idx = self.product_id_to_idx[pid_str]
        target_vector = self.image_features[target_idx].reshape(1, -1)

        # Compute Cosine Similarity between target image vector and all product image vectors
        sim_scores = cosine_similarity(target_vector, self.image_features).ravel()

        # Sort by similarity descending
        sorted_indices = np.argsort(sim_scores)[::-1]

        # Exclude target product itself
        recommended_indices = [idx for idx in sorted_indices if idx != target_idx][:top_k]

        results = []
        for rank, cand_idx in enumerate(recommended_indices, start=1):
            row = self.products_df.iloc[cand_idx].to_dict()
            score = float(sim_scores[cand_idx])
            cand_pid = str(row["product_id"]).strip().upper()

            results.append({
                "rank": rank,
                "product_id": cand_pid,
                "product_name": row["product_name"],
                "category": row["category"],
                "price": row["price"],
                "rating": row["rating"],
                "image_path": str(self.image_paths.get(cand_pid, "")),
                "similarity_score": round(score, 4),
                "recommendation_method": "CNN ResNet50 Visual Similarity"
            })

        return pd.DataFrame(results)

    def save_model(self, filepath: Union[str, Path] = "models/image_features.joblib"):
        """Saves extracted CNN feature vectors & product metadata to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "products_df": self.products_df,
            "image_features": self.image_features,
            "product_id_to_idx": self.product_id_to_idx,
            "image_paths": self.image_paths
        }
        joblib.dump(payload, filepath)
        logger.info("Saved CNNImageSimilarity artifact to %s", filepath)

    @classmethod
    def load_model(cls, filepath: Union[str, Path] = "models/image_features.joblib") -> "CNNImageSimilarity":
        """Loads fitted CNN feature vectors artifact from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model artifact not found at {filepath}")

        payload = joblib.load(filepath)
        instance = cls()
        instance.products_df = payload["products_df"]
        instance.image_features = payload["image_features"]
        instance.product_id_to_idx = payload["product_id_to_idx"]
        instance.image_paths = payload["image_paths"]
        logger.info("Loaded CNNImageSimilarity artifact from %s", filepath)
        return instance


if __name__ == "__main__":
    from src.preprocessing.pipeline import DataPreprocessor
    clean_products = DataPreprocessor().run_pipeline()
    cnn_engine = CNNImageSimilarity().fit(clean_products)
    
    sample_pid = clean_products["product_id"].iloc[0]
    similar_imgs = cnn_engine.find_similar_images(sample_pid, top_k=5)
    print(f"\nVisually Similar Products for Product {sample_pid}:")
    print(similar_imgs[["rank", "product_id", "product_name", "category", "similarity_score"]])
