"""
Flask REST API Application Factory

Provides REST endpoints for the Hybrid AI Recommendation System, semantic search,
catalog browsing, and transparent Explainable AI (XAI) attribution.
"""

from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from src.api.routes import health_bp, products_bp, search_bp, recommendations_bp
from src.api.service import get_service
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> Flask:
    """Application factory for Flask REST API."""
    static_folder = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_folder), static_url_path="")
    CORS(app)

    # Register Blueprints under /api prefix
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(products_bp, url_prefix="/api")
    app.register_blueprint(search_bp, url_prefix="/api")
    app.register_blueprint(recommendations_bp, url_prefix="/api")

    # Serve static single page application frontend at root /
    @app.route("/")
    def serve_frontend():
        index_file = static_folder / "index.html"
        if index_file.exists():
            return send_from_directory(static_folder, "index.html")
        return jsonify({
            "status": "ok",
            "message": "Hybrid AI Recommendation API is running. Access API endpoints at /api/*"
        }), 200

    # Global Error Handlers
    @app.errorhandler(400)
    def bad_request_error(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "BAD_REQUEST",
                "message": getattr(e, "description", "Bad Request")
            }
        }), 400

    @app.errorhandler(404)
    def not_found_error(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": getattr(e, "description", "Resource Not Found")
            }
        }), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal Server Error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred. Technical details logged internally."
            }
        }), 500

    # Pre-initialize ML Singleton Service
    with app.app_context():
        get_service()

    return app


if __name__ == "__main__":
    app = create_app()
    logger.info("Starting Flask REST API Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
