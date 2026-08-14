"""
Step 5 Audit: Multi-Seed Evaluation, Weight Sensitivity & Ablation Analysis Script

Runs deterministic evaluation across 5 random seeds (42, 7, 21, 100, 123).
Calculates Mean, Std, Min, Max, and 95% Confidence Intervals for Precision, Recall, and NDCG.
Exports 6 visual evaluation plots to reports/figures/ (figures 15 through 20).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.preprocessing.pipeline import DataPreprocessor
from src.data.loader import DataLoader
from src.evaluation.multi_seed_evaluator import MultiSeedEvaluator, DEFAULT_EVAL_SEEDS
from src.models import DEFAULT_WEIGHTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_multi_seed_analysis(
    processed_products_path: Path = Path("data/processed/clean_products.csv"),
    interactions_path: Path = Path("data/raw/user_interactions_raw.csv"),
    figures_dir: Path = Path("reports/figures")
):
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("      MULTI-SEED EVALUATION, SENSITIVITY ANALYSIS & ABLATION STUDY")
    print("=" * 80)

    # 1. Load Datasets
    products_df = pd.read_csv(processed_products_path) if processed_products_path.exists() else DataPreprocessor().run_pipeline()
    interactions_df = pd.read_csv(interactions_path) if interactions_path.exists() else DataLoader().load_raw_csv()

    print(f" Catalog Products : {len(products_df)} items")
    print(f" Rating Logs     : {len(interactions_df)} interactions")
    print(f" Evaluation Seeds: {DEFAULT_EVAL_SEEDS}")

    evaluator = MultiSeedEvaluator(products_df=products_df, interactions_df=interactions_df, seeds=DEFAULT_EVAL_SEEDS)

    # 2. Multi-Seed Weight Sensitivity Experiments
    print("\n--- 1. HYBRID WEIGHT SENSITIVITY EXPERIMENTS (5 RANDOM SEEDS) ---")
    weight_configs = {
        "Config A (Default Balanced)": DEFAULT_WEIGHTS,
        "Config B (Collab-Heavy)": {"content": 0.15, "collaborative": 0.55, "nlp": 0.20, "image": 0.10},
        "Config C (NLP-Heavy)": {"content": 0.20, "collaborative": 0.20, "nlp": 0.50, "image": 0.10},
        "Config D (Content/Collab)": {"content": 0.30, "collaborative": 0.40, "nlp": 0.20, "image": 0.10}
    }

    weight_results_df = evaluator.run_multi_seed_evaluation(weight_configs, top_k=5)
    print(weight_results_df[["Configuration", "P@K_95CI", "R@K_95CI", "NDCG@K_95CI"]].to_string(index=False))

    # Plot 15: Multi-Seed Metric Means with Error Bars
    fig, ax = plt.subplots(figsize=(8, 4))
    configs = weight_results_df["Configuration"].tolist()
    p_means = weight_results_df["P@K_Mean"].values
    p_stds = weight_results_df["P@K_Std"].values

    ax.errorbar(configs, p_means, yerr=p_stds, fmt='o-', capsize=5, color='indigo', ecolor='red', elinewidth=2, capthick=2)
    ax.set_title("Hybrid Weight Sensitivity: Precision@5 Mean & Std across 5 Random Seeds")
    ax.set_ylabel("Precision@5 Mean Score")
    ax.set_ylim(0, 0.15)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig15_path = figures_dir / "15_multi_seed_metric_means_errbars.png"
    plt.savefig(fig15_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig15_path}")

    # Plot 16: Multi-Seed Weight Sensitivity Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    melt_df = weight_results_df.melt(id_vars=["Configuration"], value_vars=["P@K_Mean", "R@K_Mean", "NDCG@K_Mean"], var_name="Metric", value_name="Score")
    sns.barplot(data=melt_df, x="Configuration", y="Score", hue="Metric", palette="Blues_d", ax=ax)
    ax.set_title("Multi-Seed Weight Experiments Metric Summary (K=5)")
    ax.set_ylim(0, 0.20)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig16_path = figures_dir / "16_hybrid_weight_sensitivity_multiseed.png"
    plt.savefig(fig16_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig16_path}")

    # 3. 5-Stage Ablation Study Across Multi-Seeds
    print("\n--- 2. ABLATION STUDY ACROSS 5 RANDOM SEEDS ---")
    ablation_df = evaluator.run_ablation_study(top_k=5)
    print(ablation_df[["Configuration", "P@K_95CI", "R@K_95CI", "NDCG@K_95CI"]].to_string(index=False))

    # Plot 17: Ablation Study Multi-Seed Bar Chart
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ab_melt = ablation_df.melt(id_vars=["Configuration"], value_vars=["P@K_Mean", "R@K_Mean", "NDCG@K_Mean"], var_name="Metric", value_name="Score")
    sns.barplot(data=ab_melt, x="Configuration", y="Score", hue="Metric", palette="Purples_d", ax=ax)
    ax.set_title("Ablation Study: Mean Recommendation Metrics Across Incremental Models")
    ax.set_ylim(0, 0.20)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig17_path = figures_dir / "17_ablation_study_multiseed.png"
    plt.savefig(fig17_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig17_path}")

    # 4. Signal Toggle Analysis (Image & NLP Signals ON vs OFF)
    print("\n--- 3. IMAGE & NLP SIGNAL TOGGLE ANALYSIS ---")
    toggle_dict = evaluator.run_signal_toggle_analysis(top_k=5)
    
    img_comp_df = toggle_dict["image_comparison"]
    nlp_comp_df = toggle_dict["nlp_comparison"]

    print("\nImage Signal Comparison:")
    print(img_comp_df[["Configuration", "P@K_95CI", "R@K_95CI", "NDCG@K_95CI"]].to_string(index=False))

    print("\nNLP Signal Comparison:")
    print(nlp_comp_df[["Configuration", "P@K_95CI", "R@K_95CI", "NDCG@K_95CI"]].to_string(index=False))

    # Plot 18: Image Signal ON vs OFF Comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    img_melt = img_comp_df.melt(id_vars=["Configuration"], value_vars=["P@K_Mean", "R@K_Mean", "NDCG@K_Mean"], var_name="Metric", value_name="Score")
    sns.barplot(data=img_melt, x="Configuration", y="Score", hue="Metric", palette="crest", ax=ax)
    ax.set_title("CNN Image Signal Impact: ON vs OFF")
    ax.set_ylim(0, 0.20)
    plt.tight_layout()
    fig18_path = figures_dir / "18_image_signal_on_off_comparison.png"
    plt.savefig(fig18_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig18_path}")

    # Plot 19: NLP Signal ON vs OFF Comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    nlp_melt = nlp_comp_df.melt(id_vars=["Configuration"], value_vars=["P@K_Mean", "R@K_Mean", "NDCG@K_Mean"], var_name="Metric", value_name="Score")
    sns.barplot(data=nlp_melt, x="Configuration", y="Score", hue="Metric", palette="flare", ax=ax)
    ax.set_title("SentenceTransformers NLP Signal Impact: ON vs OFF")
    ax.set_ylim(0, 0.20)
    plt.tight_layout()
    fig19_path = figures_dir / "19_nlp_signal_on_off_comparison.png"
    plt.savefig(fig19_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig19_path}")

    # Plot 20: Comprehensive Model Precision, Recall & NDCG Metric Summary
    fig, ax = plt.subplots(figsize=(9, 4.5))
    all_summary_df = pd.concat([ablation_df, weight_results_df], ignore_index=True).drop_duplicates(subset=["Configuration"])
    summary_melt = all_summary_df.melt(id_vars=["Configuration"], value_vars=["P@K_Mean", "R@K_Mean", "NDCG@K_Mean"], var_name="Metric", value_name="Score")
    sns.barplot(data=summary_melt, x="Configuration", y="Score", hue="Metric", palette="Set2", ax=ax)
    ax.set_title("Final Multi-Seed Metric Overview Across All Evaluated Configurations")
    ax.set_ylim(0, 0.20)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig20_path = figures_dir / "20_model_precision_recall_ndcg_summary.png"
    plt.savefig(fig20_path, dpi=300)
    plt.close()
    print(f" Saved plot: {fig20_path}")

    print("=" * 80 + "\n")
    logger.info("Multi-seed evaluation complete. All figures (15-20) generated.")


if __name__ == "__main__":
    run_multi_seed_analysis()
