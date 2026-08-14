"""
Unit Tests for System & Image Quality Auditor Utility
"""

import numpy as np
from src.utils.image_audit import SystemAuditor


def test_system_auditor_dataset_statistics():
    auditor = SystemAuditor()
    stats = auditor.audit_dataset_statistics()

    assert stats["is_synthetic"] is True
    assert stats["num_products"] == 140
    assert stats["num_users"] == 40
    assert stats["num_interactions"] == 600
    assert "interaction_density" in stats
    assert stats["interaction_density"] > 0.0


def test_system_auditor_image_hashes():
    auditor = SystemAuditor()
    hashes = auditor.audit_image_hashes_and_duplicates()

    assert hashes["total_images_scanned"] >= 140
    assert hashes["unique_image_hashes"] > 0
    assert "image_dimensions" in hashes


def test_calculate_category_bias():
    auditor = SystemAuditor()
    # Dummy feature vectors: 2 for Electronics, 2 for Apparel
    feats = np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.9, 0.1]
    ])
    cats = ["Electronics", "Electronics", "Apparel", "Apparel"]

    bias = auditor.calculate_category_bias(feats, cats)
    assert bias["intra_category_mean"] > bias["inter_category_mean"]
    assert bias["category_bias_delta"] > 0.0
