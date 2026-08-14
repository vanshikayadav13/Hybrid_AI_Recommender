"""
Unit Tests for Data Preprocessing Pipeline Module
"""

import pandas as pd
from pathlib import Path
from src.data.loader import DataLoader
from src.preprocessing.pipeline import DataPreprocessor


def test_text_cleaning():
    raw_html = "  <h1>Wireless Earbuds</h1> <br> <p>High Quality</p>  "
    clean = DataPreprocessor.clean_text(raw_html)
    assert clean == "Wireless Earbuds High Quality"


def test_preprocessing_pipeline(sample_raw_df: pd.DataFrame, tmp_data_dir: Path):
    preprocessor = DataPreprocessor(processed_dir=tmp_data_dir / "processed")
    
    # First map schema
    mapped_df = DataLoader.map_schema(sample_raw_df)
    
    # Process
    clean_df = preprocessor.process(mapped_df)
    
    # 1. Check deduplication (P001 had duplicate)
    assert len(clean_df) == 3
    assert clean_df["product_id"].tolist() == ["P001", "P002", "P003"]
    
    # 2. Check HTML removal from P001 title
    p001_name = clean_df.loc[clean_df["product_id"] == "P001", "product_name"].values[0]
    assert "<p>" not in p001_name
    assert "Wireless Bluetooth Earbuds Best Quality" in p001_name
    
    # 3. Check missing category imputation for P002
    p002_cat = clean_df.loc[clean_df["product_id"] == "P002", "category"].values[0]
    assert p002_cat == "Uncategorized"
    
    # 4. Check missing description imputation for P002
    p002_desc = clean_df.loc[clean_df["product_id"] == "P002", "description"].values[0]
    assert "No description available" in p002_desc
    
    # 5. Check price clamp for P002 (negative price -12 -> 0.0)
    p002_price = clean_df.loc[clean_df["product_id"] == "P002", "price"].values[0]
    assert p002_price == 0.0
    
    # 6. Check rating clamp for P002 (invalid rating 6.2 -> 5.0)
    p002_rating = clean_df.loc[clean_df["product_id"] == "P002", "rating"].values[0]
    assert p002_rating == 5.0
