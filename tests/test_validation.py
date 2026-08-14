"""
Unit Tests for Data Validation Module
"""

import pandas as pd
from src.data.validation import DataValidator


def test_validator_pass():
    clean_df = pd.DataFrame([
        {
            "product_id": "P101",
            "product_name": "Keyboard",
            "category": "Electronics",
            "description": "Mechanical keyboard",
            "price": 79.99,
            "rating": 4.5,
            "image_url": "https://img.com/p101.jpg"
        }
    ])
    
    validator = DataValidator()
    results = validator.validate_dataset(clean_df)
    
    assert results["passed"] is True
    assert results["missing_product_ids"] == 0
    assert results["duplicate_product_ids"] == 0
    assert results["invalid_prices"] == 0
    assert results["invalid_ratings"] == 0


def test_validator_detects_errors():
    invalid_df = pd.DataFrame([
        {
            "product_id": "P101",
            "product_name": "Keyboard A",
            "category": "Electronics",
            "description": "",
            "price": -10.0,  # invalid price
            "rating": 7.0,   # invalid rating
            "image_url": "invalid_url"
        },
        {
            "product_id": "P101",  # duplicate ID
            "product_name": "Keyboard B",
            "category": "Electronics",
            "description": "Valid desc",
            "price": 50.0,
            "rating": 4.0,
            "image_url": "https://img.com/b.jpg"
        }
    ])
    
    validator = DataValidator()
    results = validator.validate_dataset(invalid_df)
    
    assert results["passed"] is False
    assert results["duplicate_product_ids"] == 1
    assert results["invalid_prices"] == 1
    assert results["invalid_ratings"] == 1
    assert results["empty_descriptions"] == 1
