"""
Step 4: Hybrid Recommendation Engine & Explainable AI (XAI) Analysis Script

1. Image Similarity Quality Investigation
2. Hybrid Recommender Multi-Scenario Execution (Known User, Cold-Start User, Query, Product Context)
3. Transparent XAI Score Contribution Breakdown
4. Weight Experimentation
5. Ablation Study
6. Diagnostic Plot Generation (reports/figures/11..14)
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
from src.models import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    NLPSemanticSearch,
    CNNImageSimilarity,
    HybridRecommender,
    ExplainableAI,
    DEFAULT_WEIGHTS
)
from src.evaluation.metrics import train_test_split_interactions, precision_at_k, recall_at_k, ndcg_at_k
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_step4_analysis(
    processed_products_path: Path = Path("data/processed/clean_products.csv"),
    interactions_path: Path = Path("data/raw/user_interactions_raw.csv"),
    models_dir: Path = Path("models"),
    figures_dir: Path = Path("reports/figures")
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("       STEP 4: HYBRID RECOMMENDATION ENGINE & EXPLAINABLE AI (XAI)")
    print("=" * 80)

    # 1. Load Datasets & Fitted Models
    products_df = pd.read_csv(processed_products_path) if processed_products_path.exists() else DataPreprocessor().run_pipeline()
    interactions_df = pd.read_csv(interactions_path) if interactions_path.exists() else DataLoader().load_raw_csv()

    print(f" Loaded Products Catalog : {len(products_df)} items")
    print(f" Loaded Interactions Log : {len(interactions_df)} ratings")

    cb_model = ContentBasedRecommender.load_model(models_dir / "content_based.joblib")
    cf_model = CollaborativeRecommender.load_model(models_dir / "collaborative.joblib")
    nlp_model = NLPSemanticSearch.load_model(models_dir / "nlp_search.joblib")
    img_model = CNNImageSimilarity.load_model(models_dir / "image_features.joblib")

    # 2. PART 2 — Image Similarity Quality Investigation
    print("\n--- 1. INVESTIGATING CNN IMAGE SIMILARITY QUALITY ---")
    sim_matrix = cnn_model_sim_matrix = img_model.image_features @ img_model.image_features.T
    
    intra_cat_sims = []
    inter_cat_sims = []

    pids = products_df["product_id"].tolist()
    cats = products_df["category"].tolist()

    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            s = float(sim_matrix[i, j])
            if cats[i].lower() == cats[j].lower():
                intra_cat_sims.append(s)
            else:
                inter_cat_sims.append(s)

    intra_mean = np.mean(intra_cat_sims) if intra_cat_sims else 0.0
    inter_mean = np.mean(inter_cat_sims) if inter_cat_sims else 0.0

    print(f" Average Intra-Category Similarity (Same Category) : {intra_mean:.4f}")
    print(f" Average Inter-Category Similarity (Diff Category) : {inter_mean:.4f}")
    print(" Limitation Analysis: Synthetic images share category-coded background colors.")
    print(" Reasoned Decision: Setting initial image_weight = 0.15 to balance visual signals.")

    # Plot 1: Image Similarity Investigation Distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(intra_cat_sims, color="blue", label=f"Intra-Category (Mean: {intra_mean:.2f})", kde=True, ax=ax, alpha=0.5)
    sns.histplot(inter_cat_sims, color="orange", label=f"Inter-Category (Mean: {inter_mean:.2f})", kde=True, ax=ax, alpha=0.5)
    ax.set_title("ResNet50 Visual Similarity: Intra-Category vs Inter-Category")
    ax.set_xlabel("Cosine Similarity Score")
    ax.legend()
    plt.tight_layout()
    fig1_path = figures_dir / "11_image_similarity_investigation.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig1_path}")

    # 3. Initialize Unified Hybrid Recommender
    hybrid = HybridRecommender(
        content_model=cb_model,
        collaborative_model=cf_model,
        nlp_model=nlp_model,
        image_model=img_model,
        products_df=products_df,
        weights=DEFAULT_WEIGHTS
    )
    hybrid.save_model(models_dir / "hybrid_recommender.joblib")

    # 4. Run Multi-Scenario Recommendations with XAI Explanations
    sample_pid = products_df["product_id"].iloc[0]
    sample_uid = cf_model.user_ids[0]

    print(f"\n--- 2. HYBRID SCENARIO A: KNOWN USER ({sample_uid}) + PRODUCT ({sample_pid}) + QUERY ---")
    recs_a = hybrid.hybrid_recommend(
        user_id=sample_uid,
        product_id=sample_pid,
        query="comfortable headphones for gaming",
        top_k=5
    )
    print(recs_a[["rank", "product_id", "product_name", "final_hybrid_score", "explanation"]].to_string(index=False))

    print(f"\n--- 3. HYBRID SCENARIO B: NEW COLD-START USER + QUERY ---")
    recs_b = hybrid.hybrid_recommend(
        user_id="USER_NEW_999",
        query="stylish denim jacket",
        top_k=5
    )
    print(recs_b[["rank", "product_id", "product_name", "final_hybrid_score", "explanation"]].to_string(index=False))

    # Plot 2: XAI Score Contribution Breakdown for Top Recommendation
    fig, ax = plt.subplots(figsize=(8, 4))
    top_row = recs_a.iloc[0]
    contrib_data = pd.DataFrame([
        {"Signal": "Content", "Contribution": top_row["content_contribution"]},
        {"Signal": "Collaborative", "Contribution": top_row["collaborative_contribution"]},
        {"Signal": "NLP Semantic", "Contribution": top_row["nlp_contribution"]},
        {"Signal": "CNN Image", "Contribution": top_row["image_contribution"]}
    ])

    sns.barplot(data=contrib_data, x="Signal", y="Contribution", palette="viridis", ax=ax)
    ax.set_title(f"XAI Score Contribution Breakdown for Top Item '{top_row['product_id']}' (Total: {top_row['final_hybrid_score']:.4f})")
    ax.set_ylim(0, 0.5)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h + 0.01),
                        ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    fig2_path = figures_dir / "14_xai_score_contributions.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig2_path}")

    # 5. Weight Experimentation
    print("\n--- 4. HYBRID WEIGHT CONFIGURATION EXPERIMENTS ---")
    weight_configs = {
        "Balanced (Default)": {"content": 0.25, "collaborative": 0.35, "nlp": 0.25, "image": 0.15},
        "Collaborative-Heavy": {"content": 0.15, "collaborative": 0.55, "nlp": 0.20, "image": 0.10},
        "NLP-Heavy": {"content": 0.20, "collaborative": 0.20, "nlp": 0.50, "image": 0.10}
    }

    train_df, test_df = train_test_split_interactions(interactions_df, test_ratio=0.2, seed=42)
    eval_cf = CollaborativeRecommender().fit(train_df, products_df)

    weight_exp_results = []
    for cfg_name, w_dict in weight_configs.items():
        exp_hybrid = HybridRecommender(
            content_model=cb_model,
            collaborative_model=eval_cf,
            nlp_model=nlp_model,
            image_model=img_model,
            products_df=products_df,
            weights=w_dict
        )

        p5_list, r5_list, n5_list = [], [], []
        relevant_test = test_df[test_df["user_rating"] >= 3.0].groupby("user_id")["product_id"].apply(set).to_dict()
        
        for u_id, rel_items in relevant_test.items():
            res = exp_hybrid.hybrid_recommend(user_id=u_id, top_k=5)
            rec_items = res["product_id"].tolist() if not res.empty else []
            p5_list.append(precision_at_k(rec_items, rel_items, k=5))
            r5_list.append(recall_at_k(rec_items, rel_items, k=5))
            n5_list.append(ndcg_at_k(rec_items, rel_items, k=5))

        weight_exp_results.append({
            "Configuration": cfg_name,
            "P@5": round(float(np.mean(p5_list)), 4),
            "R@5": round(float(np.mean(r5_list)), 4),
            "NDCG@5": round(float(np.mean(n5_list)), 4)
        })

    exp_df = pd.DataFrame(weight_exp_results)
    print(exp_df.to_string(index=False))

    # Plot 3: Weight Experiments Comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    exp_melted = pd.melt(exp_df, id_vars=["Configuration"], var_name="Metric", value_name="Score")
    sns.barplot(data=exp_melted, x="Configuration", y="Score", hue="Metric", palette="Set1", ax=ax)
    ax.set_title("Hybrid Weight Experiments Metric Comparison (Cutoff K=5)")
    ax.set_ylim(0, 0.6)
    plt.tight_layout()
    fig3_path = figures_dir / "13_hybrid_weight_experiments.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig3_path}")

    # 6. Ablation Study
    print("\n--- 5. ABLATION STUDY (INCREMENTAL MODEL SIGNAL ADDITION) ---")
    ablation_stages = [
        ("Content Only", {"content": 1.0, "collaborative": 0.0, "nlp": 0.0, "image": 0.0}),
        ("Content + Collab", {"content": 0.40, "collaborative": 0.60, "nlp": 0.0, "image": 0.0}),
        ("Content + Collab + NLP", {"content": 0.30, "collaborative": 0.40, "nlp": 0.30, "image": 0.0}),
        ("Full Hybrid", DEFAULT_WEIGHTS)
    ]

    ablation_results = []
    for stage_name, w_dict in ablation_stages:
        ab_hybrid = HybridRecommender(
            content_model=cb_model,
            collaborative_model=eval_cf,
            nlp_model=nlp_model,
            image_model=img_model,
            products_df=products_df,
            weights=w_dict
        )

        p5_list, r5_list, n5_list = [], [], []
        for u_id, rel_items in relevant_test.items():
            res = ab_hybrid.hybrid_recommend(user_id=u_id, top_k=5)
            rec_items = res["product_id"].tolist() if not res.empty else []
            p5_list.append(precision_at_k(rec_items, rel_items, k=5))
            r5_list.append(recall_at_k(rec_items, rel_items, k=5))
            n5_list.append(ndcg_at_k(rec_items, rel_items, k=5))

        ablation_results.append({
            "Ablation Stage": stage_name,
            "P@5": round(float(np.mean(p5_list)), 4),
            "R@5": round(float(np.mean(r5_list)), 4),
            "NDCG@5": round(float(np.mean(n5_list)), 4)
        })

    ablation_df = pd.DataFrame(ablation_results)
    print(ablation_df.to_string(index=False))

    # Plot 4: Ablation Study Comparison Chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ab_melted = pd.melt(ablation_df, id_vars=["Ablation Stage"], var_name="Metric", value_name="Score")
    sns.barplot(data=ab_melted, x="Ablation Stage", y="Score", hue="Metric", palette="Blues_d", ax=ax)
    ax.set_title("Ablation Study: Recommendation Accuracy Across Incremental Models")
    ax.set_ylim(0, 0.6)
    plt.tight_layout()
    fig4_path = figures_dir / "12_hybrid_ablation_study.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig4_path}")

    print("=" * 80 + "\n")
    logger.info("Step 4 Analysis complete. All artifacts and figures exported.")


if __name__ == "__main__":
    run_step4_analysis()
