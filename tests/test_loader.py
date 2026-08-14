"""
Unit Tests for Data Loader & Schema Mapping Module
"""

import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader, STANDARD_SCHEMA, COLUMN_MAPPINGS


def test_schema_mapping():
    raw_data = pd.DataFrame({
        "asin": ["A100"],
        "title": ["Sample Product"],
        "categories": ["Tech"],
        "overall": [4.5],
        "imUrl": ["http://img.com/a.jpg"]
    })
    
    mapped_df = DataLoader.map_schema(raw_data)
    
    assert "product_id" in mapped_df.columns
    assert mapped_df.loc[0, "product_id"] == "A100"
    assert "product_name" in mapped_df.columns
    assert mapped_df.loc[0, "product_name"] == "Sample Product"
    assert "rating" in mapped_df.columns
    assert mapped_df.loc[0, "rating"] == 4.5
    
    # Check that all standard schema columns exist
    for col in STANDARD_SCHEMA:
        assert col in mapped_df.columns


def test_synthetic_data_generation(tmp_data_dir: Path):
    loader = DataLoader(raw_data_dir=tmp_data_dir / "raw")
    prod_df, inter_df = loader.generate_and_save_synthetic_data(n_products=20, n_users=5, n_interactions=30)
    
    assert len(prod_df) >= 20
    assert len(inter_df) == 30
    assert (tmp_data_dir / "raw" / "products_raw.csv").exists()
    assert (tmp_data_dir / "raw" / "user_interactions_raw.csv").exists()
