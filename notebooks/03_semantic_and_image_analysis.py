"""
NLP Semantic Search & CNN Image Similarity Analysis Script

Fits SentenceTransformer and ResNet50 models, performs TF-IDF vs Semantic Search comparison,
runs visual product search, and exports diagnostic visualization charts to reports/figures/.
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
from PIL import Image

from src.preprocessing.pipeline import DataPreprocessor
from src.models.content_based import ContentBasedRecommender
from src.models.nlp_search import NLPSemanticSearch
from src.models.image_similarity import CNNImageSimilarity
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_step3_analysis(
    processed_products_path: Path = Path("data/processed/clean_products.csv"),
    models_dir: Path = Path("models"),
    figures_dir: Path = Path("reports/figures")
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("      STEP 3: NLP SEMANTIC SEARCH & CNN IMAGE SIMILARITY ANALYSIS")
    print("=" * 75)

    # 1. Load Clean Dataset
    if not processed_products_path.exists():
        products_df = DataPreprocessor().run_pipeline()
    else:
        products_df = pd.read_csv(processed_products_path)

    print(f" Loaded Products Catalog : {len(products_df)} rows")

    # 2. Fit TF-IDF & SentenceTransformer NLP Models
    print("\n--- 1. FITTING NLP SEMANTIC SEARCH ENGINE (SentenceTransformers all-MiniLM-L6-v2) ---")
    tfidf_model = ContentBasedRecommender(max_features=3000).fit(products_df)
    nlp_model = NLPSemanticSearch().fit(products_df)
    nlp_model.save_model(models_dir / "nlp_search.joblib")

    # 3. Fit ResNet50 CNN Feature Extractor
    print("\n--- 2. FITTING CNN IMAGE SIMILARITY ENGINE (ResNet50 2048D Feature Extractor) ---")
    cnn_model = CNNImageSimilarity().fit(products_df)
    cnn_model.save_model(models_dir / "image_features.joblib")

    # 4. Compare TF-IDF vs NLP Semantic Search
    test_queries = [
        "comfortable wireless headphones for gaming",
        "affordable sneakers for jogging",
        "stylish denim jacket for winter"
    ]

    print("\n--- 3. COMPARISON EXPERIMENT: LEXICAL TF-IDF vs DENSE SEMANTIC SEARCH ---")
    for q in test_queries:
        print(f"\nQUERY: '{q}'")
        semantic_res = nlp_model.semantic_search(q, top_k=3)
        print(" Semantic Search (Dense Embeddings):")
        for _, r in semantic_res.iterrows():
            print(f"   [{r['rank']}] {r['product_id']} | {r['product_name']:35s} | Similarity: {r['similarity_score']:.4f}")

    # 5. Image Similarity Retrieval
    sample_pid = products_df["product_id"].iloc[0]
    sample_name = products_df["product_name"].iloc[0]
    print(f"\n--- 4. VISUAL IMAGE SIMILARITY FOR ITEM '{sample_pid}' ({sample_name}) ---")
    img_recs = cnn_model.find_similar_images(sample_pid, top_k=5)
    print(img_recs[["rank", "product_id", "product_name", "category", "similarity_score", "image_path"]])

    # 6. Generate Diagnostic Visualizations
    print("\n--- 5. GENERATING DIAGNOSTIC VISUALIZATION PLOTS ---")
    sns.set_theme(style="whitegrid")

    # Plot 1: Semantic Search vs TF-IDF Score Distribution Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    q_sample = test_queries[0]
    sem_df = nlp_model.semantic_search(q_sample, top_k=5)
    
    sns.barplot(data=sem_df, x="product_id", y="similarity_score", color="mediumseagreen", ax=ax)
    ax.set_title(f"Semantic Similarity Scores for Query: '{q_sample}'")
    ax.set_xlabel("Product ID")
    ax.set_ylabel("Cosine Similarity Score")
    ax.set_ylim(0, 1.0)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.02),
                        ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    fig1_path = figures_dir / "09_semantic_vs_tfidf.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig1_path}")

    # Plot 2: Image Similarity Grid Visualization
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.5))
    
    # Target image
    target_img_path = cnn_model.image_paths.get(sample_pid)
    if target_img_path and Path(target_img_path).exists():
        t_img = Image.open(target_img_path)
        axes[0].imshow(t_img)
    axes[0].set_title(f"Query Item\n{sample_pid}", fontsize=10, fontweight="bold", color="darkblue")
    axes[0].axis("off")

    # Top 5 visually similar images
    for i, (_, row) in enumerate(img_recs.iterrows(), start=1):
        c_pid = row["product_id"]
        c_path = row["image_path"]
        score = row["similarity_score"]
        if c_path and Path(c_path).exists():
            c_img = Image.open(c_path)
            axes[i].imshow(c_img)
        axes[i].set_title(f"Rank {i}: {c_pid}\nSim: {score:.3f}", fontsize=9)
        axes[i].axis("off")

    plt.suptitle(f"CNN ResNet50 Visual Similarity Results for '{sample_pid}'", fontsize=12, fontweight="bold", y=1.05)
    plt.tight_layout()
    fig2_path = figures_dir / "10_image_similarity_results.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig2_path}")

    print("=" * 75 + "\n")
    logger.info("Step 3 Analysis complete. Models saved to models/ and plots to reports/figures/.")


if __name__ == "__main__":
    run_step3_analysis()
