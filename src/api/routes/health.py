"""Health Check Blueprint"""
from flask import Blueprint, jsonify
from src.api.service import get_service

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """GET /api/health - Returns API & Model Service Health Status."""
    service = get_service()
    return jsonify({
        "success": True,
        "status": "ok",
        "service": "Hybrid AI Recommendation Engine API",
        "version": "1.0.0",
        "models_loaded": service._initialized,
        "total_catalog_items": len(service.products_df) if service._initialized else 0
    }), 200
