"""Products Blueprint"""
from pathlib import Path
from flask import Blueprint, jsonify, request, send_from_directory
from src.api.service import get_service

products_bp = Blueprint("products", __name__)


@products_bp.route("/products", methods=["GET"])
def get_products():
    """
    GET /api/products
    Query Parameters:
        - category (str, optional): Filter by category name
        - search (str, optional): Keyword search in title/description
        - page (int, optional, default=1): Page number
        - limit (int, optional, default=12): Items per page (max 100)
    """
    category = request.args.get("category", type=str)
    search_term = request.args.get("search", type=str)
    page = request.args.get("page", default=1, type=int)
    limit = request.args.get("limit", default=12, type=int)

    limit = max(1, min(limit, 100))
    page = max(1, page)

    service = get_service()
    result = service.get_products(category=category, search_term=search_term, page=page, limit=limit)

    return jsonify({
        "success": True,
        "data": result["products"],
        "pagination": result["pagination"]
    }), 200


@products_bp.route("/products/<product_id>", methods=["GET"])
def get_product_details(product_id: str):
    """
    GET /api/products/<product_id>
    Returns details for a single product. 404 if not found.
    """
    service = get_service()
    product = service.get_product_by_id(product_id)

    if not product:
        return jsonify({
            "success": False,
            "error": {
                "code": "PRODUCT_NOT_FOUND",
                "message": f"Product with ID '{product_id}' was not found in the catalog."
            }
        }), 404

    return jsonify({
        "success": True,
        "product": product
    }), 200


@products_bp.route("/images/<path:filename>", methods=["GET"])
def get_product_image(filename: str):
    """
    GET /api/images/<filename>
    Serves product images from data/images directory.
    """
    images_dir = Path("data/images").resolve()
    if not (images_dir / filename).exists():
        # Fallback placeholder image or 404
        return jsonify({
            "success": False,
            "error": {
                "code": "IMAGE_NOT_FOUND",
                "message": f"Image '{filename}' not found."
            }
        }), 404
    return send_from_directory(images_dir, filename)
