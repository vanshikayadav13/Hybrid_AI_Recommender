"""API Routes Package Init"""
from src.api.routes.health import health_bp
from src.api.routes.products import products_bp
from src.api.routes.search import search_bp
from src.api.routes.recommendations import recommendations_bp

__all__ = [
    "health_bp",
    "products_bp",
    "search_bp",
    "recommendations_bp"
]
