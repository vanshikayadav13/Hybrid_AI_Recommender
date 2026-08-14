"""
Data Loader & Schema Mapping Module

Responsible for loading raw datasets, mapping column names to the standardized
e-commerce schema, and generating realistic synthetic benchmark datasets.
"""

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard target schema expected across the pipeline
STANDARD_SCHEMA = [
    "product_id",
    "product_name",
    "category",
    "description",
    "price",
    "rating",
    "image_url",
    "user_id",
    "user_rating",
]

# Common raw dataset column mappings (e.g. Amazon, Kaggle, Custom datasets)
COLUMN_MAPPINGS = {
    # Amazon dataset mappings
    "asin": "product_id",
    "title": "product_name",
    "categories": "category",
    "overall": "rating",
    "imUrl": "image_url",
    "imageURL": "image_url",
    "reviewerID": "user_id",
    # Kaggle / Generic e-commerce mappings
    "item_id": "product_id",
    "product_title": "product_name",
    "product_category": "category",
    "product_description": "description",
    "unit_price": "price",
    "cost": "price",
    "avg_rating": "rating",
    "stars": "rating",
    "cust_id": "user_id",
    "interaction_rating": "user_rating",
}


class DataLoader:
    """
    Handles data loading, schema mapping, and synthetic benchmark generation.
    """

    def __init__(self, raw_data_dir: Path = Path("data/raw")):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_csv(
        self,
        filepath: Optional[Path] = None,
        custom_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Loads raw dataset CSV and standardizes column names.

        Args:
            filepath (Optional[Path]): Path to raw CSV file. If None, checks data/raw/.
            custom_mapping (Optional[Dict[str, str]]): Additional column name mappings.

        Returns:
            pd.DataFrame: DataFrame mapped towards the standard schema.
        """
        if filepath is None:
            csv_files = list(self.raw_data_dir.glob("*.csv"))
            if not csv_files:
                logger.warning("No raw CSV files found in %s. Generating synthetic data.", self.raw_data_dir)
                return self.generate_and_save_synthetic_data()[0]
            filepath = csv_files[0]

        logger.info("Loading raw dataset from %s", filepath)
        df = pd.read_csv(filepath)

        df = self.map_schema(df, custom_mapping)
        return df

    @staticmethod
    def map_schema(df: pd.DataFrame, custom_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Maps raw DataFrame column names to standard schema.

        Args:
            df (pd.DataFrame): Raw DataFrame.
            custom_mapping (Optional[Dict[str, str]]): Custom mapping dictionary.

        Returns:
            pd.DataFrame: DataFrame with standardized column names.
        """
        mapping = COLUMN_MAPPINGS.copy()
        if custom_mapping:
            mapping.update(custom_mapping)

        # Rename existing columns if match found
        rename_dict = {col: mapping[col] for col in df.columns if col in mapping}
        df = df.rename(columns=rename_dict)
        logger.info("Renamed columns: %s", rename_dict)

        # Ensure all standard columns exist (fill missing ones with NaN)
        for col in STANDARD_SCHEMA:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def generate_and_save_synthetic_data(
        self,
        n_products: int = 150,
        n_users: int = 40,
        n_interactions: int = 600,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates realistic synthetic e-commerce product and user interaction datasets.

        Args:
            n_products (int): Number of products to generate.
            n_users (int): Number of unique users.
            n_interactions (int): Number of user-product interaction records.
            seed (int): Random seed for reproducibility.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (products_df, interactions_df)
        """
        random.seed(seed)
        np.random.seed(seed)

        categories = {
            "Electronics": [
                ("Noise-Canceling Wireless Headphones", "Premium Bluetooth over-ear headphones with active noise cancellation and 30-hour battery life."),
                ("Ultra-HD 4K Smart Monitor", "32-inch 4K IPS display monitor with HDR10 support and USB-C connectivity."),
                ("Ergonomic Mechanical Keyboard", "RGB backlit tactile mechanical gaming keyboard with customizable macro keys."),
                ("High-Speed Portable SSD 1TB", "Compact external solid-state drive with 1050 MB/s transfer speed and rugged body."),
                ("Smart Fitness Watch Series X", "Waterproof smartwatch featuring heart rate monitoring, GPS tracking, and OLED display.")
            ],
            "Apparel": [
                ("Classic Unisex Cotton Hoodie", "Soft fleece lined pullover hoodie available in multiple neutral colors."),
                ("Breathable Running Shoes", "Lightweight mesh athletic sneakers designed for endurance running and training."),
                ("Slim-Fit Denim Jacket", "Timeless vintage denim jacket with durable stitching and dual chest pockets."),
                ("Thermal Compression Leggings", "High-waisted workout leggings with moisture-wicking technology and side pockets.")
            ],
            "Home & Kitchen": [
                ("Automatic Espresso Machine", "Compact 15-bar pump espresso maker with integrated milk frother wand."),
                ("Stainless Steel Cookware Set", "10-piece non-stick induction compatible pan and pot set."),
                ("Robotic Vacuum Cleaner", "Smart Wi-Fi connected robot vacuum with self-charging and laser mapping navigation.")
            ],
            "Books": [
                ("Mastering Machine Learning with Python", "A comprehensive guide to practical data science, neural networks, and AI algorithms."),
                ("Designing Data-Intensive Applications", "Essential handbook for building scalable, reliable, and maintainable software systems.")
            ],
            "Beauty & Personal Care": [
                ("Hydrating Facial Cleanser", "Gentle foaming face wash infused with hyaluronic acid and essential ceramides."),
                ("Organic Argan Oil Hair Mask", "Deep conditioning hair repair treatment enriched with vitamin E and botanical oils.")
            ]
        }

        products = []
        prod_counter = 101

        category_names = list(categories.keys())

        for i in range(n_products):
            cat = category_names[i % len(category_names)]
            templates = categories[cat]
            base_name, base_desc = templates[i % len(templates)]
            
            p_id = f"PROD_{prod_counter:04d}"
            name = f"{base_name} Model {chr(65 + (i % 26))}" if i >= len(templates) else base_name
            desc = f"{base_desc} High quality item designed for daily use."
            price = round(float(np.random.uniform(12.50, 499.99)), 2)
            rating = round(float(np.random.uniform(2.5, 5.0)), 1)
            img_url = f"https://images.example.com/products/{p_id.lower()}.jpg"

            products.append({
                "product_id": p_id,
                "product_name": name,
                "category": cat,
                "description": desc,
                "price": price,
                "rating": rating,
                "image_url": img_url,
                "user_id": np.nan,
                "user_rating": np.nan
            })
            prod_counter += 1

        # Introduce a couple intentional duplicates/edge cases for preprocessor & validation testing
        products.append(products[0].copy())  # Exact duplicate product_id
        products.append({
            "product_id": "PROD_9999",
            "product_name": "  Damaged Item with HTML <br> tag  ",
            "category": "Electronics",
            "description": "Short description missing info.",
            "price": -15.0,  # Invalid price
            "rating": 6.5,    # Invalid rating (> 5)
            "image_url": "invalid_url_format",
            "user_id": np.nan,
            "user_rating": np.nan
        })

        products_df = pd.DataFrame(products)

        # Save synthetic products
        prod_path = self.raw_data_dir / "products_raw.csv"
        products_df.to_csv(prod_path, index=False)
        logger.info("Saved synthetic raw products dataset to %s (%d rows)", prod_path, len(products_df))

        # Generate User Interactions
        user_ids = [f"USER_{u:03d}" for u in range(1, n_users + 1)]
        product_ids = [p["product_id"] for p in products[:-2]]  # clean subset

        interactions = []
        for _ in range(n_interactions):
            u_id = random.choice(user_ids)
            p_id = random.choice(product_ids)
            u_rating = round(float(random.choice([1.0, 2.0, 3.0, 4.0, 5.0])), 1)
            interactions.append({
                "user_id": u_id,
                "product_id": p_id,
                "user_rating": u_rating
            })

        interactions_df = pd.DataFrame(interactions)
        inter_path = self.raw_data_dir / "user_interactions_raw.csv"
        interactions_df.to_csv(inter_path, index=False)
        logger.info("Saved synthetic user interactions dataset to %s (%d rows)", inter_path, len(interactions_df))

        return products_df, interactions_df


if __name__ == "__main__":
    loader = DataLoader()
    raw_df = loader.load_raw_csv()
    print("Loaded Dataset Shape:", raw_df.shape)
    print("Dataset Columns:", raw_df.columns.tolist())
    print(raw_df.head(3))
