"""
Pytest Fixtures configuration.
Provides sample raw datasets, edge cases, and temporary paths for test isolation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """
    Returns a sample raw dataframe containing clean data, missing values, duplicates, and edge cases.
    """
    data = [
        {
            "item_id": "P001",
            "product_title": "  Wireless Bluetooth Earbuds <p>Best Quality</p> ",
            "product_category": "electronics",
            "product_description": "Crystal clear sound with noise reduction.",
            "unit_price": 49.99,
            "stars": 4.5,
            "imageURL": "http://images.example.com/p001.jpg"
        },
        {
            "item_id": "P001",  # Duplicate ID
            "product_title": "  Wireless Bluetooth Earbuds <p>Best Quality</p> ",
            "product_category": "electronics",
            "product_description": "Crystal clear sound with noise reduction.",
            "unit_price": 49.99,
            "stars": 4.5,
            "imageURL": "http://images.example.com/p001.jpg"
        },
        {
            "item_id": "P002",
            "product_title": "Cotton Graphic T-Shirt",
            "product_category": np.nan,  # Missing category
            "product_description": "",      # Empty description
            "unit_price": -12.0,            # Negative price
            "stars": 6.2,                   # Invalid rating > 5
            "imageURL": "invalid_path"
        },
        {
            "item_id": "P003",
            "product_title": "Stainless Steel Water Bottle",
            "product_category": "Home & Kitchen",
            "product_description": "1 Litre insulated water flask.",
            "unit_price": np.nan,           # Missing price
            "stars": np.nan,                # Missing rating
            "imageURL": "https://images.example.com/p003.jpg"
        }
    ]
    return pd.DataFrame(data)


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """
    Provides a temporary directory structure for data testing.
    """
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return tmp_path
