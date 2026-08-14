"""
Data Preprocessing Pipeline Module

Applies clean, reusable preprocessing transformations to standard raw e-commerce data:
- Missing value imputation
- Product deduplication
- Text normalization & HTML cleaning
- Numerical validation & type conversion
- File export to data/processed/
"""

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.data.loader import DataLoader, STANDARD_SCHEMA
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Production-quality Data Preprocessing Pipeline for Product Catalog & Interaction Data.
    """

    def __init__(
        self,
        processed_dir: Path = Path("data/processed"),
        default_category: str = "Uncategorized",
        default_description: str = "No description available.",
        default_price: float = 0.0,
        default_rating: float = 3.5
    ):
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.default_category = default_category
        self.default_description = default_description
        self.default_price = default_price
        self.default_rating = default_rating

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Strips whitespace, removes HTML tags, and replaces extra spaces.

        Args:
            text (str): Input text string.

        Returns:
            str: Cleaned text.
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", str(text))
        # Collapse multiple whitespaces
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate rows based on product_id and duplicate product titles.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Deduplicated DataFrame.
        """
        initial_count = len(df)
        
        # Drop duplicates by product_id
        if "product_id" in df.columns:
            df = df.drop_duplicates(subset=["product_id"], keep="first")

        # Drop exact duplicate product names if valid
        if "product_name" in df.columns:
            df = df.drop_duplicates(subset=["product_name"], keep="first")

        final_count = len(df)
        logger.info("Deduplication removed %d duplicate rows (from %d to %d)", initial_count - final_count, initial_count, final_count)
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values with sensible defaults.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Imputed DataFrame.
        """
        df = df.copy()

        # Text fields
        df["product_name"] = df["product_name"].apply(self.clean_text)
        df["product_name"] = df["product_name"].replace("", "Unnamed Product")

        df["category"] = df["category"].apply(self.clean_text)
        df["category"] = df["category"].replace("", self.default_category)

        df["description"] = df["description"].apply(self.clean_text)
        df["description"] = df["description"].replace("", self.default_description)

        df["image_url"] = df["image_url"].astype(str).str.strip()
        df["image_url"] = df["image_url"].replace(["nan", "None", ""], "https://images.example.com/placeholder.jpg")

        # Numerical fields conversion & imputation
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        median_price = df["price"].loc[df["price"] > 0].median()
        fill_price = median_price if not pd.isna(median_price) else self.default_price
        df["price"] = df["price"].fillna(fill_price)

        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["rating"] = df["rating"].fillna(self.default_rating)

        logger.info("Missing value imputation completed.")
        return df

    def handle_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates ranges: price >= 0.0, 0.0 <= rating <= 5.0.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = df.copy()

        # Fix invalid negative prices
        invalid_prices = (df["price"] < 0)
        if invalid_prices.any():
            logger.warning("Found %d negative prices. Clamping to 0.0.", invalid_prices.sum())
            df.loc[invalid_prices, "price"] = 0.0

        # Clamp ratings to [0.0, 5.0]
        df["rating"] = df["rating"].clip(lower=0.0, upper=5.0)

        return df

    def normalize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies final string formatting and category tree normalization.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Text normalized DataFrame.
        """
        df = df.copy()

        # Standardize product_id format
        df["product_id"] = df["product_id"].astype(str).str.strip().str.upper()

        # Format categories nicely (Title Case)
        df["category"] = df["category"].str.title()

        return df

    def process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the complete preprocessing pipeline on a raw DataFrame.

        Args:
            raw_df (pd.DataFrame): Raw input DataFrame.

        Returns:
            pd.DataFrame: Processed clean product catalog.
        """
        logger.info("Starting data preprocessing pipeline...")
        df = raw_df.copy()

        # Step 1: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 2: Impute missing values
        df = self.handle_missing_values(df)

        # Step 3: Handle invalid numeric ranges
        df = self.handle_invalid_values(df)

        # Step 4: Normalize text fields
        df = self.normalize_text_fields(df)

        # Reorder columns according to STANDARD_SCHEMA
        cols_to_keep = [c for c in STANDARD_SCHEMA if c in df.columns]
        df = df[cols_to_keep]

        logger.info("Preprocessing complete. Final shape: %s", df.shape)
        return df

    def run_pipeline(
        self,
        raw_filepath: Optional[Path] = None,
        output_filename: str = "clean_products.csv"
    ) -> pd.DataFrame:
        """
        Loads raw dataset, runs preprocessing pipeline, and saves clean output CSV.

        Args:
            raw_filepath (Optional[Path]): Path to raw data.
            output_filename (str): Output CSV filename in data/processed/.

        Returns:
            pd.DataFrame: Clean processed DataFrame.
        """
        loader = DataLoader()
        raw_df = loader.load_raw_csv(filepath=raw_filepath)

        clean_df = self.process(raw_df)

        output_path = self.processed_dir / output_filename
        clean_df.to_csv(output_path, index=False)
        logger.info("Successfully saved clean dataset to %s", output_path)

        return clean_df


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    clean_df = preprocessor.run_pipeline()
    print("Clean Dataset Sample:")
    print(clean_df.head())
