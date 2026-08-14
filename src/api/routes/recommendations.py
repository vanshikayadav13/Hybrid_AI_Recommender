"""Hybrid Recommendations & XAI Blueprint"""
from flask import Blueprint, jsonify, request
from src.api.service import get_service

recommendations_bp = Blueprint("recommendations", __name__)


@recommendations_bp.route("/recommendations/product/<product_id>", methods=["GET"])
def get_product_recommendations(product_id: str):
    """
    GET /api/recommendations/product/<product_id>?top_k=10
    Returns similar/hybrid recommendations for a reference product.
    """
    top_k = request.args.get("top_k", default=10, type=int)
    top_k = max(1, min(top_k, 50))

    service = get_service()
    product = service.get_product_by_id(product_id)
    if not product:
        return jsonify({
            "success": False,
            "error": {
                "code": "PRODUCT_NOT_FOUND",
                "message": f"Product '{product_id}' not found."
            }
        }), 404

    recs = service.get_recommendations_hybrid(product_id=product_id, top_k=top_k)
    return jsonify({
        "success": True,
        "product_id": product_id,
        "recommendations": recs
    }), 200


@recommendations_bp.route("/recommendations/user/<user_id>", methods=["GET"])
def get_user_recommendations(user_id: str):
    """
    GET /api/recommendations/user/<user_id>?top_k=10
    Returns personalized hybrid recommendations for a target user.
    Handles cold-start users gracefully.
    """
    top_k = request.args.get("top_k", default=10, type=int)
    top_k = max(1, min(top_k, 50))

    service = get_service()
    recs = service.get_recommendations_hybrid(user_id=user_id, top_k=top_k)
    return jsonify({
        "success": True,
        "user_id": user_id,
        "recommendations": recs
    }), 200


@recommendations_bp.route("/recommendations/hybrid", methods=["POST"])
def get_hybrid_recommendations():
    """
    POST /api/recommendations/hybrid
    Body (JSON):
        {
            "user_id": "USER_001",
            "product_id": "PROD_0101",
            "query": "wireless headphones",
            "top_k": 10,
            "weights": {"content": 0.25, "collaborative": 0.35, "nlp": 0.25, "image": 0.15}
        }
    """
    data = request.get_json(silent=True) or {}

    user_id = data.get("user_id")
    product_id = data.get("product_id")
    query = data.get("query")
    weights = data.get("weights")
    top_k = data.get("top_k", 10)

    try:
        top_k = int(top_k)
        top_k = max(1, min(top_k, 50))
    except (ValueError, TypeError):
        top_k = 10

    service = get_service()

    if weights and isinstance(weights, dict):
        try:
            service.hybrid_engine.validate_weights(weights)
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": {
                    "code": "INVALID_WEIGHTS",
                    "message": str(e)
                }
            }), 400

    recs = service.get_recommendations_hybrid(
        user_id=user_id,
        product_id=product_id,
        query=query,
        top_k=top_k,
        weights=weights
    )

    return jsonify({
        "success": True,
        "parameters": {
            "user_id": user_id,
            "product_id": product_id,
            "query": query
        },
        "recommendations": recs
    }), 200


@recommendations_bp.route("/recommendations/<product_id>/explanation", methods=["GET"])
def get_recommendation_explanation(product_id: str):
    """
    GET /api/recommendations/<product_id>/explanation?reference_product_id=...&user_id=...&query=...
    Returns XAI score attribution breakdown for a recommendation item.
    """
    ref_pid = request.args.get("reference_product_id")
    user_id = request.args.get("user_id")
    query = request.args.get("query")

    service = get_service()
    recs = service.get_recommendations_hybrid(
        user_id=user_id,
        product_id=ref_pid,
        query=query,
        top_k=20
    )

    # Find target product_id in results
    target_rec = None
    for r in recs:
        if str(r.get("product_id")).strip().upper() == str(product_id).strip().upper():
            target_rec = r
            break

    if not target_rec:
        # Fallback explanation if item wasn't in top 20 candidates
        prod = service.get_product_by_id(product_id)
        if not prod:
            return jsonify({
                "success": False,
                "error": {
                    "code": "PRODUCT_NOT_FOUND",
                    "message": f"Product '{product_id}' not found."
                }
            }), 404

        target_rec = {
            "product_id": product_id,
            "product_name": prod["product_name"],
            "final_hybrid_score": 0.50,
            "content_contribution": 0.25,
            "collaborative_contribution": 0.25,
            "nlp_contribution": 0.0,
            "image_contribution": 0.0,
            "explanation": "Recommended based on overall catalog category matching and popularity."
        }

    return jsonify({
        "success": True,
        "explanation": {
            "product_id": target_rec["product_id"],
            "product_name": target_rec["product_name"],
            "final_hybrid_score": target_rec.get("final_hybrid_score", 0.0),
            "score_contributions": {
                "content": target_rec.get("content_contribution", 0.0),
                "collaborative": target_rec.get("collaborative_contribution", 0.0),
                "nlp": target_rec.get("nlp_contribution", 0.0),
                "image": target_rec.get("image_contribution", 0.0)
            },
            "reasoning": target_rec.get("explanation", ""),
            "notice": "Transparent score attribution model based on weighted signal combination."
        }
    }), 200
