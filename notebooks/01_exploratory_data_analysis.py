"""
Exploratory Data Analysis (EDA) Script

Performs thorough statistical analysis and visual diagnostics on processed product data:
1. Dataset Shape & Data Types
2. Missing Values Analysis
3. Product & Category Distribution
4. Price Distribution & Quantiles
5. Rating Distribution
6. User-Product Interaction Density (if available)
7. Exports publication-ready charts to reports/figures/
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI compatibility
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.preprocessing.pipeline import DataPreprocessor
from src.data.loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_eda(
    clean_csv_path: Path = Path("data/processed/clean_products.csv"),
    figures_dir: Path = Path("reports/figures")
):
    """
    Executes EDA calculations and saves visual plots.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data (trigger pipeline if clean CSV doesn't exist yet)
    if not clean_csv_path.exists():
        logger.info("Clean dataset not found at %s. Triggering preprocessing pipeline.", clean_csv_path)
        preprocessor = DataPreprocessor()
        df = preprocessor.run_pipeline()
    else:
        logger.info("Loading cleaned dataset from %s", clean_csv_path)
        df = pd.read_csv(clean_csv_path)

    # Set visualization theme
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.titlesize": 14})

    print("\n" + "=" * 70)
    print("           EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 70)

    # 1. Dataset Overview
    print("\n--- 1. DATASET OVERVIEW ---")
    print(f"Total Rows (Products): {df.shape[0]}")
    print(f"Total Columns        : {df.shape[1]}")
    print("\nColumns & Data Types:")
    for col in df.columns:
        print(f"  - {col:20s}: {str(df[col].dtype):10s} (Non-null count: {df[col].count()})")

    # 2. Missing Values & Duplicates
    print("\n--- 2. MISSING VALUES & DUPLICATES ANALYSIS ---")
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_series, "Missing %": missing_pct.round(2)})
    print(missing_df)

    dup_ids = df["product_id"].duplicated().sum() if "product_id" in df.columns else 0
    print(f"\nDuplicate Product IDs: {dup_ids}")

    # Plot 1: Missing Values Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    missing_pct.plot(kind="bar", color="skyblue", edgecolor="black", ax=ax)
    ax.set_title("Missing Values Percentage per Column")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig1_path = figures_dir / "01_missing_values.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig1_path}")

    # 3. Product Category Distribution
    print("\n--- 3. CATEGORY DISTRIBUTION ---")
    cat_counts = df["category"].value_counts()
    print(cat_counts)

    # Plot 2: Category Distribution Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=cat_counts.values, y=cat_counts.index, hue=cat_counts.index, palette="viridis", legend=False, ax=ax)
    ax.set_title("Product Count by Category")
    ax.set_xlabel("Number of Products")
    ax.set_ylabel("Category")
    for i, v in enumerate(cat_counts.values):
        ax.text(v + 0.5, i, str(v), va="center", fontweight="bold")
    plt.tight_layout()
    fig2_path = figures_dir / "02_category_distribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig2_path}")

    # 4. Price Distribution Statistics
    print("\n--- 4. PRICE DISTRIBUTION STATISTICS ---")
    prices = df["price"].dropna()
    print(f"Mean Price   : ${prices.mean():.2f}")
    print(f"Median Price : ${prices.median():.2f}")
    print(f"Min Price    : ${prices.min():.2f}")
    print(f"Max Price    : ${prices.max():.2f}")
    print(f"Std Dev      : ${prices.std():.2f}")
    print(f"25th %-ile   : ${prices.quantile(0.25):.2f}")
    print(f"75th %-ile   : ${prices.quantile(0.75):.2f}")

    # Plot 3: Price Distribution Histogram & Boxplot
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": (0.2, 0.8)}
    )
    sns.boxplot(x=prices, ax=ax_box, color="coral")
    ax_box.set(xlabel="")
    ax_box.set_title("Price Distribution Analysis ($)")

    sns.histplot(prices, kde=True, ax=ax_hist, color="teal", bins=20)
    ax_hist.set_xlabel("Price ($)")
    ax_hist.set_ylabel("Product Count")
    plt.tight_layout()
    fig3_path = figures_dir / "03_price_distribution.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig3_path}")

    # 5. Rating Distribution Statistics
    print("\n--- 5. RATING DISTRIBUTION STATISTICS ---")
    ratings = df["rating"].dropna()
    print(f"Mean Rating   : {ratings.mean():.2f} / 5.0")
    print(f"Median Rating : {ratings.median():.2f} / 5.0")
    print(f"Min Rating    : {ratings.min():.2f}")
    print(f"Max Rating    : {ratings.max():.2f}")

    # Plot 4: Rating Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(ratings, bins=10, kde=True, color="gold", edgecolor="darkgoldenrod", ax=ax)
    ax.set_title("Product Rating Distribution (0.0 to 5.0)")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Product Count")
    plt.tight_layout()
    fig4_path = figures_dir / "04_rating_distribution.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig4_path}")

    # 6. User Interaction Data Check (if available)
    inter_path = Path("data/raw/user_interactions_raw.csv")
    if inter_path.exists():
        print("\n--- 6. USER INTERACTION DATASET OVERVIEW ---")
        inter_df = pd.read_csv(inter_path)
        print(f"Total Interactions : {len(inter_df)}")
        print(f"Unique Users       : {inter_df['user_id'].nunique()}")
        print(f"Unique Products    : {inter_df['product_id'].nunique()}")
        print(f"Sparsity Level     : {1.0 - (len(inter_df) / (inter_df['user_id'].nunique() * inter_df['product_id'].nunique())):.4f}")

    print("=" * 70 + "\n")
    logger.info("EDA completed successfully. All figures saved to %s", figures_dir)


if __name__ == "__main__":
    run_eda()
