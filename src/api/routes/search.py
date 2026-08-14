"""NLP Semantic Search Blueprint"""
from flask import Blueprint, jsonify, request
from src.api.service import get_service

search_bp = Blueprint("search", __name__)


@search_bp.route("/search", methods=["POST"])
def search_products():
    """
    POST /api/search
    Body (JSON):
        {
            "query": "affordable running shoes",
            "top_k": 10
        }
    Returns SentenceTransformer semantic vector search results.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": {
                "code": "INVALID_JSON",
                "message": "Request body must be a valid JSON object."
            }
        }), 400

    query = data.get("query")
    if not query or not isinstance(query, str) or not query.strip():
        return jsonify({
            "success": False,
            "error": {
                "code": "MISSING_QUERY",
                "message": "Field 'query' is required and cannot be empty."
            }
        }), 400

    top_k = data.get("top_k", 10)
    try:
        top_k = int(top_k)
        top_k = max(1, min(top_k, 50))
    except (ValueError, TypeError):
        top_k = 10

    service = get_service()
    results = service.search_semantic(query=query, top_k=top_k)

    return jsonify({
        "success": True,
        "query": query,
        "total_results": len(results),
        "results": results
    }), 200
