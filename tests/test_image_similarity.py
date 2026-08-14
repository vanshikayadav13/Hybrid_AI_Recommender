"""
Unit Tests for ResNet50 CNN Image Similarity Module
"""

import pytest
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.image_utils import safe_load_image, create_synthetic_image
from src.models.image_similarity import CNNImageSimilarity


@pytest.fixture
def test_images_dir(tmp_path: Path) -> Path:
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. RGB Image
    rgb_img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    rgb_img.save(img_dir / "rgb.jpg")
    
    # 2. RGBA Image
    rgba_img = Image.new("RGBA", (150, 150), color=(0, 255, 0, 128))
    rgba_img.save(img_dir / "rgba.png")
    
    # 3. Grayscale Image
    gray_img = Image.new("L", (200, 200), color=128)
    gray_img.save(img_dir / "gray.jpg")
    
    # 4. Corrupted Image file
    bad_file = img_dir / "corrupt.jpg"
    bad_file.write_text("Not an image file content")
    
    return img_dir


@pytest.fixture
def sample_catalog_df() -> pd.DataFrame:
    data = [
        {"product_id": "P101", "product_name": "Headphones", "category": "Electronics", "price": 100.0, "rating": 4.5},
        {"product_id": "P102", "product_name": "Earbuds", "category": "Electronics", "price": 50.0, "rating": 4.0},
        {"product_id": "P103", "product_name": "Running Shoes", "category": "Apparel", "price": 80.0, "rating": 4.2},
    ]
    return pd.DataFrame(data)


def test_safe_load_image_formats(test_images_dir: Path):
    # Test RGB loading
    rgb = safe_load_image(test_images_dir / "rgb.jpg")
    assert rgb.mode == "RGB"
    
    # Test RGBA conversion to RGB
    rgba = safe_load_image(test_images_dir / "rgba.png")
    assert rgba.mode == "RGB"
    
    # Test Grayscale conversion to RGB
    gray = safe_load_image(test_images_dir / "gray.jpg")
    assert gray.mode == "RGB"
    
    # Test missing image handling
    with pytest.raises(FileNotFoundError):
        safe_load_image(test_images_dir / "non_existent.jpg")
        
    # Test corrupted image handling
    with pytest.raises(ValueError):
        safe_load_image(test_images_dir / "corrupt.jpg")


def test_resnet_feature_extraction(test_images_dir: Path):
    cnn = CNNImageSimilarity()
    vector = cnn.extract_image_features(test_images_dir / "rgb.jpg")
    
    # ResNet50 feature vector shape must be 2048
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (2048,)
    # Vector should be unit-normalized
    norm = np.linalg.norm(vector)
    assert abs(norm - 1.0) < 1e-4


def test_image_similarity_search(sample_catalog_df: pd.DataFrame, tmp_path: Path):
    img_dir = tmp_path / "images"
    cnn = CNNImageSimilarity().fit(sample_catalog_df, image_dir=img_dir)
    
    # Search similar images for P101
    results = cnn.find_similar_images("P101", top_k=2)
    assert len(results) == 2
    assert "product_id" in results.columns
    assert "similarity_score" in results.columns
    
    # Self-exclusion check: P101 must NOT be in recommendations
    assert "P101" not in results["product_id"].tolist()
    
    # Similarity ordering check
    scores = results["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_cnn_model_persistence(sample_catalog_df: pd.DataFrame, tmp_path: Path):
    img_dir = tmp_path / "images"
    cnn = CNNImageSimilarity().fit(sample_catalog_df, image_dir=img_dir)
    
    save_file = tmp_path / "image_features.joblib"
    cnn.save_model(save_file)
    
    # Load model back
    loaded_cnn = CNNImageSimilarity.load_model(save_file)
    assert loaded_cnn.image_features.shape == (3, 2048)
    results = loaded_cnn.find_similar_images("P101", top_k=1)
    assert len(results) == 1
