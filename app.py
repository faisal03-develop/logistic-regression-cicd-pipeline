"""
Flask REST API for the Credit Risk Classifier.

Endpoints
---------
GET  /           → welcome message + version
GET  /health     → health check (model readiness)
GET  /metrics    → model performance metrics
POST /predict    → credit default prediction
"""

import os
import traceback
from datetime import datetime

from flask import Flask, jsonify, render_template, request

from src.predict import predict, get_metrics, registry
from src.preprocess import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def success(data: dict, status: int = 200):
        return jsonify({"status": "success", **data}), status

    def error(message: str, status: int = 400, details: str | None = None):
        body = {"status": "error", "message": message}
        if details:
            body["details"] = details
        return jsonify(body), status

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/")
    def index():
        return success({
            "service": "Credit Risk Classifier API",
            "version": "1.0.0",
            "description": (
                "Predicts whether a loan applicant is likely to default "
                "using Logistic Regression trained from scratch."
            ),
            "endpoints": {
                "GET  /":         "This welcome message",
                "GET  /health":   "Model health check",
                "GET  /metrics":  "Model performance metrics",
                "POST /predict":  "Predict credit default risk",
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    @app.get("/app")
    def frontend_app():
        return render_template("index.html")

    @app.get("/health")
    def health():
        is_ready = registry.is_ready()
        payload = {
            "healthy": is_ready,
            "model_loaded": is_ready,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        status_code = 200 if is_ready else 503
        return jsonify({"status": "success" if is_ready else "error", **payload}), status_code

    @app.get("/metrics")
    def metrics():
        try:
            m = get_metrics()
            if not m:
                return error("Metrics file not found. Train the model first.", 503)
            return success({"metrics": m})
        except RuntimeError as e:
            return error(str(e), 503)
        except Exception as e:
            return error("Failed to load metrics.", 500, traceback.format_exc())

    @app.post("/predict")
    def predict_endpoint():
        # --- Parse JSON body ---
        if not request.is_json:
            return error(
                "Request must have Content-Type: application/json", 415
            )

        body = request.get_json(silent=True)
        if body is None:
            return error("Invalid JSON payload.", 400)

        # --- Validate required fields ---
        missing = [f for f in FEATURE_COLUMNS if f not in body]
        if missing:
            return error(
                f"Missing required feature(s): {missing}",
                400,
                f"Required features: {FEATURE_COLUMNS}",
            )

        # --- Validate numeric types ---
        type_errors = []
        for field in FEATURE_COLUMNS:
            try:
                float(body[field])
            except (TypeError, ValueError):
                type_errors.append(field)
        if type_errors:
            return error(
                f"Non-numeric value(s) for feature(s): {type_errors}", 400
            )

        # --- Validate value ranges ---
        range_checks = {
            "age": (18, 100),
            "credit_score": (300, 850),
            "debt_to_income_ratio": (0.0, 1.0),
            "has_mortgage": (0, 1),
            "num_open_accounts": (0, 50),
            "num_credit_inquiries": (0, 50),
            "loan_tenure_months": (1, 360),
        }
        range_errors = []
        for field, (lo, hi) in range_checks.items():
            val = float(body[field])
            if not (lo <= val <= hi):
                range_errors.append(f"{field} must be in [{lo}, {hi}], got {val}")
        if range_errors:
            return error("Value(s) out of valid range.", 400, "; ".join(range_errors))

        # --- Run inference ---
        try:
            result = predict(body)
            return success({
                "input": {f: body[f] for f in FEATURE_COLUMNS},
                "result": result,
            })
        except RuntimeError as e:
            return error(str(e), 503)
        except Exception as e:
            return error("Prediction failed.", 500, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Error handlers
    # -----------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(e):
        return error(f"Endpoint not found: {request.path}", 404)

    @app.errorhandler(405)
    def method_not_allowed(e):
        return error(
            f"Method {request.method} not allowed on {request.path}", 405
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n[App] Starting Credit Risk Classifier API on port {port}")
    print(f"[App] Debug mode: {debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)
