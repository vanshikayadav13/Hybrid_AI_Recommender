"""
Recommendation Model Analysis, Evaluation & Visualization Script

Executes Content-Based and Collaborative Filtering models, evaluates ranking metrics,
generates comparison reports, and saves visual diagnostic figures to reports/figures/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.preprocessing.pipeline import DataPreprocessor
from src.data.loader import DataLoader
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative_filtering import CollaborativeRecommender
from src.models.cold_start import RecommendationEngine, ColdStartHandler
from src.evaluation.metrics import evaluate_recommender, train_test_split_interactions
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_analysis(
    processed_products_path: Path = Path("data/processed/clean_products.csv"),
    interactions_path: Path = Path("data/raw/user_interactions_raw.csv"),
    models_dir: Path = Path("models"),
    figures_dir: Path = Path("reports/figures")
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("        STEP 2: RECOMMENDATION MODEL ANALYSIS & EVALUATION")
    print("=" * 75)

    # 1. Load Datasets
    if not processed_products_path.exists():
        logger.info("Processed data not found. Running preprocessor pipeline...")
        products_df = DataPreprocessor().run_pipeline()
    else:
        products_df = pd.read_csv(processed_products_path)

    if not interactions_path.exists():
        logger.info("Interactions data not found. Running data loader synthetic generator...")
        DataLoader().generate_and_save_synthetic_data()
    
    interactions_df = pd.read_csv(interactions_path)

    print(f" Loaded Products Catalog : {len(products_df)} rows")
    print(f" Loaded Interactions Log : {len(interactions_df)} rows")

    # 2. Fit & Save Content-Based Recommender
    print("\n--- 1. FITTING CONTENT-BASED RECOMMENDER (TF-IDF + COSINE SIMILARITY) ---")
    cb_model = ContentBasedRecommender(max_features=3000).fit(products_df)
    cb_model.save_model(models_dir / "content_based.joblib")

    # 3. Fit & Save Collaborative Filtering Recommender
    print("\n--- 2. FITTING COLLABORATIVE FILTERING RECOMMENDER (ITEM-ITEM) ---")
    cf_model = CollaborativeRecommender().fit(interactions_df, products_df)
    cf_model.save_model(models_dir / "collaborative.joblib")

    # 4. Initialize Unified Recommendation Engine Facade
    engine = RecommendationEngine(
        content_model=cb_model,
        collaborative_model=cf_model,
        products_df=products_df
    )

    # 5. Sample Content-Based Recommendations
    sample_pid = products_df["product_id"].iloc[0]
    sample_name = products_df["product_name"].iloc[0]
    print(f"\n--- 3. SAMPLE CONTENT-BASED RECOMMENDATIONS FOR ITEM '{sample_pid}' ({sample_name}) ---")
    cb_recs = engine.recommend_for_product(sample_pid, top_k=5)
    print(cb_recs[["rank", "product_id", "product_name", "category", "similarity_score", "explanation"]])

    # 6. Sample Collaborative Recommendations
    sample_uid = cf_model.user_ids[0]
    print(f"\n--- 4. SAMPLE COLLABORATIVE RECOMMENDATIONS FOR USER '{sample_uid}' ---")
    cf_recs = engine.recommend_for_user(sample_uid, top_k=5)
    print(cf_recs[["rank", "product_id", "product_name", "category", "predicted_score", "explanation"]])

    # 7. Sample Cold Start Strategy for Unknown User
    unknown_uid = "USER_UNKNOWN_999"
    print(f"\n--- 5. COLD-START RECOMMENDATIONS FOR NEW USER '{unknown_uid}' ---")
    cold_recs = engine.recommend_for_user(unknown_uid, top_k=5)
    print(cold_recs[["rank", "product_id", "product_name", "category", "rating", "explanation"]])

    # 8. Model Evaluation Framework (Train/Test Split)
    print("\n--- 6. EVALUATING RECOMMENDATION METRICS (PRECISION@K, RECALL@K, NDCG@K) ---")
    train_df, test_df = train_test_split_interactions(interactions_df, test_ratio=0.2, seed=42)
    eval_cf = CollaborativeRecommender().fit(train_df, products_df)
    eval_metrics_df = evaluate_recommender(eval_cf, test_df, k_list=[5, 10], threshold_rating=3.0)
    print("\nEvaluation Metrics Table:")
    print(eval_metrics_df.to_string(index=False))

    # 9. Generate & Save Diagnostic Figures
    print("\n--- 7. GENERATING DIAGNOSTIC VISUALIZATION PLOTS ---")
    sns.set_theme(style="whitegrid")

    # Plot 1: Content Similarity Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sim_flat = cb_model.similarity_matrix.flatten()
    sim_off_diag = sim_flat[sim_flat < 0.999]  # Exclude self-similarity 1.0
    sns.histplot(sim_off_diag, bins=30, kde=True, color="mediumpurple", ax=ax)
    ax.set_title("Content-Based Pairwise Cosine Similarity Distribution")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Pair Count")
    plt.tight_layout()
    fig1_path = figures_dir / "05_content_similarity_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig1_path}")

    # Plot 2: User-Item Interaction Heatmap (Subset)
    fig, ax = plt.subplots(figsize=(10, 6))
    subset_matrix = cf_model.user_item_matrix.iloc[:15, :20]
    sns.heatmap(subset_matrix, cmap="YlGnBu", cbar_kws={"label": "User Rating"}, annot=True, fmt=".1f", ax=ax)
    ax.set_title("User-Item Rating Interaction Matrix (15 Users x 20 Products)")
    ax.set_xlabel("Product ID")
    ax.set_ylabel("User ID")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig2_path = figures_dir / "06_interaction_matrix_heatmap.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig2_path}")

    # Plot 3: Item-Item Similarity Matrix Heatmap (Subset)
    fig, ax = plt.subplots(figsize=(9, 7))
    subset_sim = cf_model.item_similarity_df.iloc[:15, :15]
    sns.heatmap(subset_sim, cmap="magma", cbar_kws={"label": "Cosine Similarity"}, annot=False, ax=ax)
    ax.set_title("Item-Item Collaborative Similarity Matrix (15 x 15 Subset)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig3_path = figures_dir / "07_collaborative_item_similarity.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig3_path}")

    # Plot 4: Evaluation Metrics Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_melted = pd.melt(eval_metrics_df, id_vars=["Metric Cutoff (K)"], var_name="Metric", value_name="Score")
    sns.barplot(data=metrics_melted, x="Metric Cutoff (K)", y="Score", hue="Metric", palette="Set2", ax=ax)
    ax.set_title("Recommendation Accuracy Metrics (Precision@K, Recall@K, NDCG@K)")
    ax.set_ylim(0, 1.0)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height + 0.02),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig4_path = figures_dir / "08_evaluation_metrics.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig4_path}")

    print("=" * 75 + "\n")
    logger.info("Step 2 Analysis complete. Artifacts saved to models/ and figures to reports/figures/.")


if __name__ == "__main__":
    run_analysis()
