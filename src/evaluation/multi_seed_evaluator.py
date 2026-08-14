"""
Multi-Seed Evaluation & Sensitivity Analysis Module

Provides deterministic multi-seed evaluation framework to measure:
1. Metric stability (Mean, Std, Min, Max, 95% Confidence Intervals) across random train/test seeds
2. Controlled hybrid weight sensitivity analysis
3. 5-Stage Ablation study (Content -> Collab -> Content+Collab -> +NLP -> Full Hybrid)
4. Isolated Image Signal (ON vs OFF) and NLP Signal (ON vs OFF) impact analysis
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.models import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    NLPSemanticSearch,
    CNNImageSimilarity,
    HybridRecommender,
    DEFAULT_WEIGHTS
)
from src.evaluation.metrics import (
    train_test_split_interactions,
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_EVAL_SEEDS = [42, 7, 21, 100, 123]


class MultiSeedEvaluator:
    """
    Executes robust multi-seed evaluation across recommendation models and signal configurations.
    """

    def __init__(
        self,
        products_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        seeds: Optional[List[int]] = None
    ):
        self.products_df = products_df.copy()
        self.interactions_df = interactions_df.copy()
        
        if seeds is not None and len(seeds) == 0:
            raise ValueError("Seeds list cannot be empty.")
            
        self.seeds = seeds if seeds is not None else DEFAULT_EVAL_SEEDS

        # Fit static content, nlp, and image models once (content-based models don't rely on user ratings)
        self.cb_model = ContentBasedRecommender().fit(self.products_df)
        self.nlp_model = NLPSemanticSearch().fit(self.products_df)
        self.img_model = CNNImageSimilarity().fit(self.products_df)

    def evaluate_model_on_seed(
        self,
        seed: int,
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluates collaborative and hybrid recommender on a single random train/test split seed.
        """
        train_df, test_df = train_test_split_interactions(self.interactions_df, test_ratio=0.2, seed=seed)
        eval_cf = CollaborativeRecommender().fit(train_df, self.products_df)

        hybrid = HybridRecommender(
            content_model=self.cb_model,
            collaborative_model=eval_cf,
            nlp_model=self.nlp_model,
            image_model=self.img_model,
            products_df=self.products_df,
            weights=weights or DEFAULT_WEIGHTS
        )

        relevant_test = test_df[test_df["user_rating"] >= 3.0].groupby("user_id")["product_id"].apply(set).to_dict()
        
        p_list, r_list, n_list = [], [], []
        for u_id, rel_items in relevant_test.items():
            recs = hybrid.hybrid_recommend(user_id=u_id, top_k=top_k, weights=weights)
            rec_items = recs["product_id"].tolist() if not recs.empty else []
            p_list.append(precision_at_k(rec_items, rel_items, k=top_k))
            r_list.append(recall_at_k(rec_items, rel_items, k=top_k))
            n_list.append(ndcg_at_k(rec_items, rel_items, k=top_k))

        return {
            "precision": float(np.mean(p_list)) if p_list else 0.0,
            "recall": float(np.mean(r_list)) if r_list else 0.0,
            "ndcg": float(np.mean(n_list)) if n_list else 0.0
        }

    @staticmethod
    def calculate_summary_stats(metric_values: List[float]) -> Dict[str, float]:
        """
        Calculates Mean, Standard Deviation, Min, Max, and 95% Confidence Interval.
        """
        if not metric_values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci_95": 0.0}

        arr = np.array(metric_values, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        
        # 95% CI margin = 1.96 * (std / sqrt(n))
        n = len(arr)
        ci_95 = float(1.96 * (std_val / np.sqrt(n))) if n > 1 else 0.0

        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "ci_95": round(ci_95, 4)
        }

    def run_multi_seed_evaluation(
        self,
        configurations: Dict[str, Dict[str, float]],
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Runs evaluation across all random seeds for specified weight configurations.
        """
        results_rows = []

        for config_name, w_dict in configurations.items():
            p_seeds, r_seeds, n_seeds = [], [], []
            for seed in self.seeds:
                res = self.evaluate_model_on_seed(seed=seed, weights=w_dict, top_k=top_k)
                p_seeds.append(res["precision"])
                r_seeds.append(res["recall"])
                n_seeds.append(res["ndcg"])

            p_stats = self.calculate_summary_stats(p_seeds)
            r_stats = self.calculate_summary_stats(r_seeds)
            n_stats = self.calculate_summary_stats(n_seeds)

            results_rows.append({
                "Configuration": config_name,
                "P@K_Mean": p_stats["mean"],
                "P@K_Std": p_stats["std"],
                "P@K_95CI": f"{p_stats['mean']:.4f} ± {p_stats['ci_95']:.4f}",
                "R@K_Mean": r_stats["mean"],
                "R@K_Std": r_stats["std"],
                "R@K_95CI": f"{r_stats['mean']:.4f} ± {r_stats['ci_95']:.4f}",
                "NDCG@K_Mean": n_stats["mean"],
                "NDCG@K_Std": n_stats["std"],
                "NDCG@K_95CI": f"{n_stats['mean']:.4f} ± {n_stats['ci_95']:.4f}",
                "Cutoff_K": top_k
            })

        return pd.DataFrame(results_rows)

    def run_ablation_study(self, top_k: int = 5) -> pd.DataFrame:
        """
        Executes 5-stage ablation study across all evaluation seeds.
        """
        ablation_stages = {
            "1. Content Only": {"content": 1.0, "collaborative": 0.0, "nlp": 0.0, "image": 0.0},
            "2. Collaborative Only": {"content": 0.0, "collaborative": 1.0, "nlp": 0.0, "image": 0.0},
            "3. Content + Collaborative": {"content": 0.40, "collaborative": 0.60, "nlp": 0.0, "image": 0.0},
            "4. Content + Collab + NLP": {"content": 0.30, "collaborative": 0.40, "nlp": 0.30, "image": 0.0},
            "5. Full Hybrid (All Signals)": DEFAULT_WEIGHTS
        }
        return self.run_multi_seed_evaluation(ablation_stages, top_k=top_k)

    def run_signal_toggle_analysis(self, top_k: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Executes isolated Image (ON vs OFF) and NLP (ON vs OFF) signal impact analysis.
        """
        image_configs = {
            "Image Signal ON": {"content": 0.25, "collaborative": 0.35, "nlp": 0.25, "image": 0.15},
            "Image Signal OFF": {"content": 0.30, "collaborative": 0.40, "nlp": 0.30, "image": 0.0}
        }
        nlp_configs = {
            "NLP Signal ON": {"content": 0.25, "collaborative": 0.35, "nlp": 0.25, "image": 0.15},
            "NLP Signal OFF": {"content": 0.35, "collaborative": 0.45, "nlp": 0.0, "image": 0.20}
        }

        image_df = self.run_multi_seed_evaluation(image_configs, top_k=top_k)
        nlp_df = self.run_multi_seed_evaluation(nlp_configs, top_k=top_k)

        return {
            "image_comparison": image_df,
            "nlp_comparison": nlp_df
        }
