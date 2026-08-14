"""
Data Validation & Integrity Checker Module

Provides rigorous data quality and sanity checks for e-commerce datasets:
- Schema enforcement
- Key uniqueness checks
- Range & bounds validation (prices, ratings)
- Empty field checks
- Image URL/filepath structure checks
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.data.loader import STANDARD_SCHEMA
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates quality, completeness, and schema consistency of product datasets.
    """

    def __init__(self, required_columns: List[str] = None):
        self.required_columns = required_columns or [
            "product_id", "product_name", "category", "price", "rating"
        ]

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive data quality checks on the provided DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to inspect.

        Returns:
            Dict[str, Any]: Structured dictionary with validation statistics and errors.
        """
        results = {
            "passed": True,
            "total_rows": len(df),
            "missing_columns": [],
            "missing_product_ids": 0,
            "duplicate_product_ids": 0,
            "invalid_prices": 0,
            "invalid_ratings": 0,
            "empty_descriptions": 0,
            "invalid_image_urls": 0,
            "details": []
        }

        # 1. Schema check
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            results["passed"] = False
            results["missing_columns"] = missing_cols
            results["details"].append(f"ERROR: Missing required schema columns: {missing_cols}")

        if len(df) == 0:
            results["passed"] = False
            results["details"].append("ERROR: DataFrame is empty.")
            return results

        # 2. Missing Product IDs
        if "product_id" in df.columns:
            missing_ids = df["product_id"].isna() | (df["product_id"].astype(str).str.strip() == "")
            results["missing_product_ids"] = int(missing_ids.sum())
            if results["missing_product_ids"] > 0:
                results["passed"] = False
                results["details"].append(f"ERROR: Found {results['missing_product_ids']} missing or blank product_ids.")

        # 3. Duplicate Product IDs
        if "product_id" in df.columns:
            dup_ids = df["product_id"].duplicated()
            results["duplicate_product_ids"] = int(dup_ids.sum())
            if results["duplicate_product_ids"] > 0:
                results["passed"] = False
                results["details"].append(f"ERROR: Found {results['duplicate_product_ids']} duplicate product_ids.")

        # 4. Invalid Prices
        if "price" in df.columns:
            prices = pd.to_numeric(df["price"], errors="coerce")
            invalid_p = prices.isna() | (prices < 0)
            results["invalid_prices"] = int(invalid_p.sum())
            if results["invalid_prices"] > 0:
                results["passed"] = False
                results["details"].append(f"WARNING: Found {results['invalid_prices']} invalid or negative prices.")

        # 5. Invalid Ratings
        if "rating" in df.columns:
            ratings = pd.to_numeric(df["rating"], errors="coerce")
            invalid_r = ratings.isna() | (ratings < 0.0) | (ratings > 5.0)
            results["invalid_ratings"] = int(invalid_r.sum())
            if results["invalid_ratings"] > 0:
                results["passed"] = False
                results["details"].append(f"WARNING: Found {results['invalid_ratings']} ratings outside the [0.0, 5.0] range.")

        # 6. Empty Descriptions
        if "description" in df.columns:
            empty_desc = df["description"].isna() | (df["description"].astype(str).str.strip() == "")
            results["empty_descriptions"] = int(empty_desc.sum())
            if results["empty_descriptions"] > 0:
                results["details"].append(f"INFO: Found {results['empty_descriptions']} empty product descriptions.")

        # 7. Invalid Image URLs / Paths
        if "image_url" in df.columns:
            invalid_urls = (
                df["image_url"].isna() | 
                (~df["image_url"].astype(str).str.contains(r"^http|^\.|^/|[a-zA-Z0-9_-]+\.(?:jpg|jpeg|png|webp)", case=False, regex=True))
            )
            results["invalid_image_urls"] = int(invalid_urls.sum())
            if results["invalid_image_urls"] > 0:
                results["details"].append(f"INFO: Found {results['invalid_image_urls']} missing or malformed image paths/urls.")

        return results

    def print_report(self, df: pd.DataFrame) -> bool:
        """
        Runs validation and prints a clean formatted summary to console.

        Args:
            df (pd.DataFrame): DataFrame to validate.

        Returns:
            bool: True if validation passed with no critical errors, False otherwise.
        """
        res = self.validate_dataset(df)

        print("\n" + "=" * 60)
        print("         DATA VALIDATION & QUALITY REPORT")
        print("=" * 60)
        print(f" Status                : {'PASSED [OK]' if res['passed'] else 'FAILED [ACTION REQUIRED]'}")
        print(f" Total Rows            : {res['total_rows']}")
        print(f" Missing Product IDs   : {res['missing_product_ids']}")
        print(f" Duplicate Product IDs : {res['duplicate_product_ids']}")
        print(f" Invalid Prices        : {res['invalid_prices']}")
        print(f" Invalid Ratings       : {res['invalid_ratings']}")
        print(f" Empty Descriptions    : {res['empty_descriptions']}")
        print(f" Invalid Image URLs    : {res['invalid_image_urls']}")
        print("-" * 60)
        
        if res["details"]:
            print(" Details:")
            for msg in res["details"]:
                print(f"  - {msg}")
        else:
            print(" All integrity checks passed clean!")
            
        print("=" * 60 + "\n")

        return res["passed"]


if __name__ == "__main__":
    from src.preprocessing.pipeline import DataPreprocessor
    preprocessor = DataPreprocessor()
    clean_df = preprocessor.run_pipeline()
    validator = DataValidator()
    validator.print_report(clean_df)
